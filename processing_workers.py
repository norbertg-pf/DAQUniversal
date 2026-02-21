import queue
import time
import numpy as np
import nidaqmx
from nidaqmx import stream_readers
from nidaqmx.constants import AcquisitionType, CJCSource, ProductCategory, ThermocoupleType
import scipy.signal as signal


def _base_signal_name(sig_name: str) -> str:
    return sig_name.split("@", 1)[0] if "@" in sig_name else sig_name


def _qualified_eval_alias(sig_name: str) -> str:
    if "@" not in sig_name:
        return sig_name
    base, dev = sig_name.split("@", 1)
    safe_dev = "".join(ch if ch.isalnum() else "_" for ch in dev)
    return f"{base}__{safe_dev}"


def get_terminal_name_with_dev_prefix(task: nidaqmx.Task, terminal_name: str) -> str:
    for device in task.devices:
        if device.product_category not in [ProductCategory.C_SERIES_MODULE, ProductCategory.SCXI_MODULE]:
            return f"/{device.name}/{terminal_name}"
    raise RuntimeError("Suitable device not found in task.")

# =============================================================================
# MULTIPROCESSING WORKERS (Runs on separate CPU Cores)
# =============================================================================

def daq_read_worker(stop_event, simulate, read_rate, samples_per_read, active_ai_configs,
                    n_ai, n_ao, active_ao_signals, has_dmm, available_signals, ao_state_dict, dmm_buffer_list,
                    tdms_q, process_q):
    """ Runs on Core 2: Handles hardware communication and pulls raw arrays. """
    sample_nr = 0
    safe_timeout = (samples_per_read / read_rate) + 2.0
    tasks = []
    readers = []
    ai_grouped = []

    def build_ao_chunk():
        if n_ao <= 0:
            return np.empty((0, samples_per_read))
        ao_vals = np.array([ao_state_dict.get(sig, 0.0) for sig in active_ao_signals], dtype=np.float64)
        return np.repeat(ao_vals[:, None], samples_per_read, axis=1)

    def build_dmm_chunk():
        if not has_dmm:
            return np.empty((0, samples_per_read))
        dmm_data = np.asarray(list(dmm_buffer_list))
        del dmm_buffer_list[:]
        if len(dmm_data) == 0:
            dmm_chunk = np.zeros(samples_per_read)
        elif len(dmm_data) < samples_per_read:
            reps = samples_per_read // len(dmm_data)
            dmm_chunk = np.concatenate([np.repeat(dmm_data, reps), np.repeat(dmm_data[-1], samples_per_read - len(dmm_data) * reps)])
        else:
            idx = np.linspace(0, len(dmm_data), samples_per_read + 1, endpoint=True).astype(int)
            dmm_chunk = np.array([dmm_data[idx[i]:idx[i+1]].mean() if len(dmm_data[idx[i]:idx[i+1]]) > 0 else 0.0 for i in range(samples_per_read)])
        return dmm_chunk.reshape(1, -1)

    try:
        if simulate:
            t_wave = 0
            while not stop_event.is_set():
                t_start = time.time()
                time_arr = np.linspace(t_wave, t_wave + samples_per_read/read_rate, samples_per_read, endpoint=False)
                t_wave += samples_per_read/read_rate

                if n_ai > 0:
                    ai_data = np.random.uniform(-0.1, 0.1, (n_ai, samples_per_read))
                    ai_data[0, :] += np.sin(2 * np.pi * 50 * time_arr) * 2.0
                    for i, cfg in enumerate(active_ai_configs):
                        if cfg.get('SensorType') == "Type K":
                            ai_data[i, :] = np.random.uniform(24.5, 25.5, samples_per_read)
                    raw_ai_tdms = ai_data.copy()
                else:
                    raw_ai_tdms = np.empty((0, samples_per_read))

                ao_chunk = build_ao_chunk()
                dmm_chunk = build_dmm_chunk()
                global_time = (sample_nr + np.arange(samples_per_read)) / read_rate

                try:
                    tdms_q.put_nowait((global_time, raw_ai_tdms, ao_chunk.copy(), dmm_chunk.copy()))
                except queue.Full:
                    pass

                data_to_process = np.vstack((raw_ai_tdms, ao_chunk, dmm_chunk)) if len(available_signals) > 0 else np.empty((0, samples_per_read))
                try:
                    process_q.put_nowait((global_time, data_to_process))
                except queue.Full:
                    pass

                sample_nr += samples_per_read
                elapsed = time.time() - t_start
                if (samples_per_read / read_rate) - elapsed > 0:
                    time.sleep((samples_per_read / read_rate) - elapsed)
            return

        if n_ai > 0:
            by_terminal = {}
            for cfg in active_ai_configs:
                term = cfg.get('Terminal', '')
                dev = term.split('/', 1)[0] if '/' in term else ''
                by_terminal.setdefault(dev, []).append(cfg)

            for dev in sorted(by_terminal.keys()):
                cfgs = by_terminal[dev]
                task = nidaqmx.Task()
                for ch in cfgs:
                    channel_name = ch.get('Terminal', '<unknown channel>')
                    if ch.get('SensorType') == "Type K":
                        task.ai_channels.add_ai_thrmcpl_chan(channel_name, thermocouple_type=ThermocoupleType.K, cjc_source=CJCSource.BUILT_IN)
                    else:
                        task.ai_channels.add_ai_voltage_chan(channel_name, terminal_config=ch['Config'], min_val=ch['Range'][0], max_val=ch['Range'][1])
                task.timing.cfg_samp_clk_timing(rate=read_rate, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=int(read_rate * 10))
                reader = stream_readers.AnalogMultiChannelReader(task.in_stream)
                task.start()
                tasks.append(task)
                readers.append(reader)
                ai_grouped.append((cfgs, np.zeros((len(cfgs), samples_per_read), dtype=np.float64)))

        while not stop_event.is_set():
            try:
                if readers:
                    ai_chunks = []
                    for reader, (_, buf) in zip(readers, ai_grouped):
                        reader.read_many_sample(data=buf, number_of_samples_per_channel=samples_per_read, timeout=safe_timeout)
                        ai_chunks.append(buf.copy())
                    raw_ai_tdms = np.vstack(ai_chunks) if ai_chunks else np.empty((0, samples_per_read))
                else:
                    time.sleep(samples_per_read / read_rate)
                    raw_ai_tdms = np.empty((0, samples_per_read))
            except Exception:
                continue

            ao_chunk = build_ao_chunk()
            dmm_chunk = build_dmm_chunk()
            global_time = (sample_nr + np.arange(samples_per_read)) / read_rate

            try:
                tdms_q.put_nowait((global_time, raw_ai_tdms, ao_chunk.copy(), dmm_chunk.copy()))
            except queue.Full:
                pass

            data_to_process = np.vstack((raw_ai_tdms, ao_chunk, dmm_chunk))
            try:
                process_q.put_nowait((global_time, data_to_process))
            except queue.Full:
                pass

            sample_nr += samples_per_read

    except Exception as e:
        print(f"[ERROR] read_voltages process crashed: {e}")
    finally:
        for task in tasks:
            try:
                task.stop(); task.close()
            except Exception:
                pass
def math_processing_worker(stop_event, rate, average_samples, available_signals, 
                           hw_signals, math_signals, cfg_dict, process_q, gui_q):
    """ Runs on Core 3: Handles high-speed math, Butterworth filtering, and averaging """
    num_hw = len(hw_signals)
    num_total = len(available_signals)
    if num_total == 0: return

    filter_sos = {}
    filter_states = {}
    for sig in available_signals:
        if sig in cfg_dict and cfg_dict[sig].get("LPF_On", False):
            cutoff = cfg_dict[sig].get("LPF_Cutoff", 10.0)
            order = cfg_dict[sig].get("LPF_Order", 4)
            if cutoff < (rate / 2.0):
                filter_sos[sig] = signal.butter(order, cutoff, btype='low', fs=rate, output='sos')
                filter_states[sig] = None

    math_samps = int(rate * 0.5)
    math_buffer = np.zeros((num_total, math_samps), dtype=np.float64)

    accum_data = []
    accum_t = []
    accum_len = 0

    while not stop_event.is_set():
        try: t_chunk, data_chunk = process_q.get(timeout=0.1)
        except queue.Empty: continue

        n_new = data_chunk.shape[1]
        processed_chunk = np.zeros((num_total, n_new), dtype=np.float64)
        eval_dict = {}

        # FILTERING & SCALING
        for i, sig in enumerate(hw_signals):
            row = data_chunk[i, :]
            if sig in filter_sos:
                if filter_states[sig] is None:
                    filter_states[sig] = signal.sosfilt_zi(filter_sos[sig]) * row[0]
                row, filter_states[sig] = signal.sosfilt(filter_sos[sig], row, zi=filter_states[sig])
            
            scale = cfg_dict[sig].get("Scale", 1.0) if sig in cfg_dict else 1.0
            offset = cfg_dict[sig].get("Offset", 0.0) if sig in cfg_dict else 0.0
            processed_row = (row * scale) - offset
            processed_chunk[i, :] = processed_row
            eval_dict[sig] = processed_row
            base_name = _base_signal_name(sig)
            if base_name not in eval_dict:
                eval_dict[base_name] = processed_row
            eval_dict[_qualified_eval_alias(sig)] = processed_row

        # VIRTUAL MATH
        eval_dict['np'] = np
        for i, sig in enumerate(math_signals):
            expr = cfg_dict[sig].get("Expression", "0")
            try:
                result = eval(expr, {"__builtins__": None}, eval_dict)
                if isinstance(result, (int, float)): result = np.full(n_new, result)
                processed_chunk[num_hw + i, :] = result
            except Exception:
                processed_chunk[num_hw + i, :] = np.zeros(n_new)
            eval_dict[sig] = processed_chunk[num_hw + i, :]

        if n_new >= math_samps: math_buffer = processed_chunk[:, -math_samps:]
        else:
            math_buffer = np.roll(math_buffer, -n_new, axis=1)
            math_buffer[:, -n_new:] = processed_chunk

        # INDICATOR MATH (100ms)
        samples_100ms = int(rate * 0.1)
        latest_math_values = {}
        
        for i, sig in enumerate(available_signals):
            sig_data = math_buffer[i, :]
            if len(sig_data) == 0: continue
            
            cur_avg = np.mean(sig_data[-samples_100ms:]) if len(sig_data) > samples_100ms else np.mean(sig_data)
            rms = np.sqrt(np.mean(np.square(sig_data)))
            p2p = np.max(sig_data) - np.min(sig_data)
            centered = sig_data - np.mean(sig_data)
            crossings = np.where((centered[:-1] < 0) & (centered[1:] >= 0))[0]
            freq = (len(crossings) - 1) / (len(crossings)/rate) if len(crossings) > 1 else 0.0
                
            latest_math_values[sig] = {"Current (100ms avg)": cur_avg, "RMS": rms, "Peak-to-Peak": p2p, "Frequency": freq}

        accum_data.append(processed_chunk)
        accum_t.append(t_chunk)
        accum_len += n_new

        # DECIMATE/AVERAGE FOR GUI
        if accum_len >= average_samples:
            big_data = np.concatenate(accum_data, axis=1)
            big_t = np.concatenate(accum_t)
            
            n_points = accum_len // average_samples
            valid_len = n_points * average_samples
            
            avg_data = np.mean(big_data[:, :valid_len].reshape((num_total, n_points, average_samples)), axis=2)
            avg_t = np.mean(big_t[:valid_len].reshape((n_points, average_samples)), axis=1)

            # Dictionary structure to send back to Main Process
            chunk_dict = {sig: avg_data[i, :] for i, sig in enumerate(available_signals)}
            try: gui_q.put_nowait((avg_t, chunk_dict, latest_math_values))
            except queue.Full: pass

            rem_data = big_data[:, valid_len:]
            rem_t = big_t[valid_len:]
            accum_data = [rem_data] if rem_data.shape[1] > 0 else []
            accum_t = [rem_t] if rem_t.shape[0] > 0 else []
            accum_len = rem_data.shape[1]

# =============================================================================
# UI WIDGET: HELP DIALOGS & CLASSES
# =============================================================================
