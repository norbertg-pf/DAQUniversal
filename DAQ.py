import sys
import os
import json
import threading
import multiprocessing as mp
import queue
import socket
import numpy as np
import nidaqmx
import nidaqmx.system
from nidaqmx import stream_readers
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, LineGrouping, ProductCategory, ThermocoupleType, CJCSource

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QGridLayout,
                             QCheckBox, QFileDialog, QScrollArea,
                             QTabWidget, QComboBox, QDialog, QMessageBox, QGroupBox, QDoubleSpinBox, QFormLayout, QFrame)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread

# --- REPLACED MATPLOTLIB WITH PYQTGRAPH ---
import pyqtgraph as pg
pg.setConfigOption('background', 'w')  # Set white background to match Matplotlib
pg.setConfigOption('foreground', 'k')  # Set black axes/text

from nptdms import TdmsWriter, ChannelObject, TdmsFile
from datetime import datetime
import collections
import time
from pathlib import Path
import traceback
import re
import scipy.signal as signal  

# =============================================================================
# GOOGLE DRIVE API IMPORTS (BOTH METHODS)
# =============================================================================
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials as OAuthCredentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

# Global Constants
ALL_AI_CHANNELS = [f"AI{i}" for i in range(32)]
ALL_AO_CHANNELS = [f"AO{i}" for i in range(4)]
ALL_MATH_CHANNELS = [f"MATH{i}" for i in range(4)]
ALL_CHANNELS = ALL_AI_CHANNELS + ALL_AO_CHANNELS + ALL_MATH_CHANNELS + ["DMM"]

PLOT_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75)]

def get_terminal_name_with_dev_prefix(task: nidaqmx.Task, terminal_name: str) -> str:
    for device in task.devices:
        if device.product_category not in [ProductCategory.C_SERIES_MODULE, ProductCategory.SCXI_MODULE]:
            return f"/{device.name}/{terminal_name}"
    raise RuntimeError("Suitable device not found in task.")

# =============================================================================
# MULTIPROCESSING WORKERS (Runs on separate CPU Cores)
# =============================================================================

def daq_read_worker(stop_event, simulate, read_rate, samples_per_read, active_ai_configs, 
                    n_ai, n_ao, has_dmm, available_signals, ao_state_dict, dmm_buffer_list, 
                    tdms_q, process_q):
    """ Runs on Core 2: Handles hardware communication and pulls raw arrays. """
    sample_nr = 0
    safe_timeout = (samples_per_read / read_rate) + 2.0
    task = None
    stream_reader = None

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
                    ai_data = raw_ai_tdms = np.empty((0, samples_per_read))

                if n_ao > 0:
                    ao_vals = np.array([ao_state_dict.get(f"AO{i}", 0.0) for i in range(4) if f"AO{i}" in available_signals])
                    ao_chunk = np.repeat(ao_vals[:, None], samples_per_read, axis=1)
                else: ao_chunk = np.empty((0, samples_per_read))

                # Handle DMM from shared list
                if has_dmm: 
                    dmm_data = np.asarray(list(dmm_buffer_list))
                    del dmm_buffer_list[:] # clear buffer
                    if len(dmm_data) == 0: dmm_chunk = np.zeros(samples_per_read)
                    elif len(dmm_data) < samples_per_read:
                        dmm_chunk = np.concatenate([np.repeat(dmm_data, samples_per_read // len(dmm_data)), np.repeat(dmm_data[-1], samples_per_read - len(dmm_data)* (samples_per_read // len(dmm_data)))])
                    else:
                        idx = np.linspace(0, len(dmm_data), samples_per_read+1, endpoint=True).astype(int)
                        dmm_chunk = np.array([dmm_data[idx[i]:idx[i+1]].mean() if len(dmm_data[idx[i]:idx[i+1]]) > 0 else dmm_chunk[-1] for i in range(samples_per_read)])
                    dmm_chunk = dmm_chunk.reshape(1, -1)
                else: dmm_chunk = np.empty((0, samples_per_read))
                
                global_time = (sample_nr + np.arange(samples_per_read)) / read_rate
                
                try: tdms_q.put_nowait((global_time, raw_ai_tdms, ao_chunk.copy(), dmm_chunk.copy()))
                except queue.Full: pass 
                
                data_to_process = np.vstack((raw_ai_tdms, ao_chunk, dmm_chunk)) if len(available_signals) > 0 else np.empty((0, samples_per_read))
                try: process_q.put_nowait((global_time, data_to_process))
                except queue.Full: pass 
                
                sample_nr += samples_per_read
                elapsed = time.time() - t_start
                if (samples_per_read / read_rate) - elapsed > 0: time.sleep((samples_per_read / read_rate) - elapsed)
            return

        # Real Hardware Setup
        ai_data = np.zeros((n_ai, samples_per_read), dtype=np.float64) if n_ai > 0 else np.empty((0, samples_per_read))
        if n_ai > 0:
            task = nidaqmx.Task()
            for ch in active_ai_configs:
                if ch.get('SensorType') == "Type K": 
                    task.ai_channels.add_ai_thrmcpl_chan(ch['Terminal'], thermocouple_type=ThermocoupleType.K, cjc_source=CJCSource.BUILT_IN)
                else: 
                    task.ai_channels.add_ai_voltage_chan(ch['Terminal'], terminal_config=ch['Config'], min_val=ch['Range'][0], max_val=ch['Range'][1])
            task.timing.cfg_samp_clk_timing(rate=read_rate, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=int(read_rate * 10))
            stream_reader = stream_readers.AnalogMultiChannelReader(task.in_stream)
            task.start()

        while not stop_event.is_set():
            try:
                if task is not None:
                    stream_reader.read_many_sample(data=ai_data, number_of_samples_per_channel=samples_per_read, timeout=safe_timeout)
                    raw_ai_tdms = ai_data.copy()
                else:
                    time.sleep(samples_per_read / read_rate)
                    raw_ai_tdms = ai_data
            except Exception: continue

            if n_ao > 0:
                ao_vals = np.array([ao_state_dict.get(f"AO{i}", 0.0) for i in range(4) if f"AO{i}" in available_signals])
                ao_chunk = np.repeat(ao_vals[:, None], samples_per_read, axis=1)
            else: ao_chunk = np.empty((0, samples_per_read))

            if has_dmm: 
                dmm_data = np.asarray(list(dmm_buffer_list))
                del dmm_buffer_list[:] 
                if len(dmm_data) == 0: dmm_chunk = np.zeros(samples_per_read)
                elif len(dmm_data) < samples_per_read:
                    dmm_chunk = np.concatenate([np.repeat(dmm_data, samples_per_read // len(dmm_data)), np.repeat(dmm_data[-1], samples_per_read - len(dmm_data)* (samples_per_read // len(dmm_data)))])
                else:
                    idx = np.linspace(0, len(dmm_data), samples_per_read+1, endpoint=True).astype(int)
                    dmm_chunk = np.array([dmm_data[idx[i]:idx[i+1]].mean() if len(dmm_data[idx[i]:idx[i+1]]) > 0 else dmm_chunk[-1] for i in range(samples_per_read)])
                dmm_chunk = dmm_chunk.reshape(1, -1)
            else: dmm_chunk = np.empty((0, samples_per_read))

            global_time = (sample_nr + np.arange(samples_per_read)) / read_rate
            
            try: tdms_q.put_nowait((global_time, raw_ai_tdms, ao_chunk.copy(), dmm_chunk.copy()))
            except queue.Full: pass
            
            data_to_process = np.vstack((raw_ai_tdms, ao_chunk, dmm_chunk))
            try: process_q.put_nowait((global_time, data_to_process))
            except queue.Full: pass

            sample_nr += samples_per_read

    except Exception as e: print(f"[ERROR] read_voltages process crashed: {e}")
    finally:
        if task is not None:
            try: task.stop(); task.close()
            except: pass

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
class GDriveHelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Google Drive Upload - Help & Setup")
        self.setMinimumWidth(600)
        self.setMinimumHeight(450)
        layout = QVBoxLayout(self)
        
        text = """
        <h3 style="color: #0055a4;">How to use the Google Drive Auto-Uploader</h3>
        <b>1. The 'auth' Folder (Security)</b><br>
        The software will automatically create an <b>auth</b> folder in your project directory. 
        You must place your security keys inside this folder. Add <code>auth/</code> to your <code>.gitignore</code> so keys are never uploaded to Git.<br><br>
        <b>2. Authentication Methods</b><br>
        <b>Option A: OAuth 2.0 (User Login) - <i>Recommended for Shared PCs</i></b><br>
        - In Google Cloud Console, create an <i>OAuth Client ID (Desktop App)</i>.<br>
        - Download the JSON, rename it to exactly <b>client_secret.json</b>, and place it in the <b>auth/</b> folder.<br>
        - The first time a measurement finishes, a browser will pop up asking you to log in with your company email.<br>
        - A "Ghost Token" (<b>token.json</b>) will be saved in the auth folder. The software will securely upload files in the background as YOU for all future tests.<br><br>
        <b>Option B: Service Account (Robot)</b><br>
        - In Google Cloud Console, create a <i>Service Account</i> and download its JSON key.<br>
        - Rename it to exactly <b>credentials.json</b> and place it in the <b>auth/</b> folder.<br>
        - <i>Important:</i> You must open your Google Drive and share the Target Folder with the Robot's email address.<br><br>
        <b>3. Google Drive Target Folder Link</b><br>
        - Open your Google Drive in a web browser.<br>
        - Navigate inside the folder where you want the data saved (For Robot uploads, this <i>must</i> be a Shared Drive).<br>
        - Copy the URL from your browser's address bar and paste it into the text box.<br>
        """
        
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.RichText)
        lbl.setStyleSheet("font-size: 13px; line-height: 1.5;")
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.addWidget(lbl)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        
        layout.addWidget(scroll)
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

class MathHelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Virtual Math Channels - Help")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        
        text = """
        <h3 style="color: #0055a4;">Using Virtual Math Channels</h3>
        Math channels let you compute values in real-time using your active Analog Inputs.
        <br><br>
        <b>How to write expressions:</b><br>
        Use the EXACT raw channel names (e.g., <b>AI0</b>, <b>AI5</b>, <b>DMM</b>) in your formula. Do NOT use your custom names.
        <br><br>
        <b>Examples:</b><br>
        • Voltage Difference: <code>AI0 - AI1</code><br>
        • Power (V * I): <code>AI0 * AI2</code><br>
        • Scaling/Offset: <code>(AI3 * 100.0) + 5.5</code><br>
        • NumPy Math: <code>np.sin(AI0)</code> or <code>np.abs(AI1)</code><br>
        <br>
        <i>Note: The math evaluates AFTER the individual channel's filtering, scaling, and offsets are applied!</i>
        """
        
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.RichText)
        lbl.setStyleSheet("font-size: 13px; line-height: 1.5;")
        layout.addWidget(lbl)
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


# =============================================================================
# GOOGLE DRIVE UPLOAD THREAD 
# =============================================================================
class UploadThread(QThread):
    status_signal = pyqtSignal(str, str) 

    def __init__(self, file_path, gdrive_link, auth_method, is_export=False):
        super().__init__()
        self.file_path = file_path
        self.gdrive_link = gdrive_link
        self.auth_method = auth_method
        self.is_export = is_export

    def run(self):
        if not GOOGLE_API_AVAILABLE:
            self.status_signal.emit("Failed: Google API missing", "orange")
            return
            
        try:
            auth_dir = os.path.join(os.getcwd(), 'auth')
            os.makedirs(auth_dir, exist_ok=True)
            base_folder_id = None
            match = re.search(r'folders/([a-zA-Z0-9_-]+)', self.gdrive_link)
            if match:
                base_folder_id = match.group(1)
            else:
                match = re.search(r'id=([a-zA-Z0-9_-]+)', self.gdrive_link)
                if match: base_folder_id = match.group(1)
                else:
                    self.status_signal.emit("Failed: Invalid Link", "orange")
                    return

            self.status_signal.emit("Authenticating...", "orange")
            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            creds = None

            if self.auth_method == "OAuth 2.0 (User Login)":
                token_path = os.path.join(auth_dir, 'token.json')
                secret_path = os.path.join(auth_dir, 'client_secret.json')
                
                if os.path.exists(token_path):
                    creds = OAuthCredentials.from_authorized_user_file(token_path, SCOPES)
                if not creds or not creds.valid:
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                    else:
                        if not os.path.exists(secret_path):
                            self.status_signal.emit("Failed: auth/client_secret.json missing", "orange")
                            return
                        flow = InstalledAppFlow.from_client_secrets_file(secret_path, SCOPES)
                        creds = flow.run_local_server(port=0)
                    with open(token_path, 'w') as token:
                        token.write(creds.to_json())
            
            elif self.auth_method == "Service Account (Robot)":
                cred_path = os.path.join(auth_dir, 'credentials.json')
                if not os.path.exists(cred_path):
                    self.status_signal.emit("Failed: auth/credentials.json missing", "orange")
                    return
                creds = ServiceAccountCredentials.from_service_account_file(cred_path, scopes=SCOPES)
            else:
                self.status_signal.emit("Failed: Unknown Auth Method", "orange")
                return

            service = build('drive', 'v3', credentials=creds)
            date_str = datetime.now().strftime("%d_%m_%Y")
            
            def get_or_create_folder(folder_name, parent_id):
                query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                results = service.files().list(
                    q=query, spaces='drive', fields='nextPageToken, files(id, name)',
                    includeItemsFromAllDrives=True, supportsAllDrives=True, corpora='allDrives'
                ).execute()
                items = results.get('files', [])
                if not items:
                    file_metadata = {
                        'name': folder_name,
                        'mimeType': 'application/vnd.google-apps.folder',
                        'parents': [parent_id]
                    }
                    folder = service.files().create(body=file_metadata, fields='id', supportsAllDrives=True).execute()
                    return folder.get('id')
                else:
                    return items[0]['id']

            self.status_signal.emit(f"Resolving Cloud Folder...", "orange")
            target_folder_id = get_or_create_folder(date_str, base_folder_id)
            if self.is_export:
                target_folder_id = get_or_create_folder("export", target_folder_id)

            self.status_signal.emit(f"Uploading file...", "orange")
            file_metadata = {'name': os.path.basename(self.file_path), 'parents': [target_folder_id]}
            media = MediaFileUpload(self.file_path, resumable=True)
            
            service.files().create(body=file_metadata, media_body=media, fields='id', supportsAllDrives=True).execute()
            self.status_signal.emit("Upload OK", "green")
            
        except Exception as e:
            self.status_signal.emit(f"Upload Failed", "red")
            print(f"[GDRIVE ERROR] {e}")


# =============================================================================
# INTEGRATED ANALOG OUT PULSE CONTROL CLASSES
# =============================================================================
class PulseSignals(QObject):
    status = pyqtSignal(str, str)            
    voltage_update = pyqtSignal(list, float) 
    done = pyqtSignal()                      

class PulseWorker(QThread):
    def __init__(self, channels, park_v, pulse_v, duration_s, parent=None):
        super().__init__(parent)
        self.channels = channels
        self.park_v = float(park_v)
        self.pulse_v = float(pulse_v)
        self.duration_s = float(duration_s)
        self._stop = False
        self.signals = PulseSignals()

    def stop(self): self._stop = True

    def _write_all(self, task, value: float):
        if len(self.channels) > 1: task.write([value] * len(self.channels))
        else: task.write(value)
        self.signals.voltage_update.emit(self.channels, value)

    def run(self):
        try:
            with nidaqmx.Task() as task:
                for ch in self.channels:
                    task.ao_channels.add_ao_voltage_chan(ch, min_val=-10.0, max_val=10.0)

                self._write_all(task, self.park_v)
                if self._stop:
                    self._write_all(task, self.park_v)
                    self.signals.status.emit(f"Voltage: PARK ({self.park_v:.2f} V)", "LOW")
                    return

                self._write_all(task, self.pulse_v)
                self.signals.status.emit(f"Voltage: PULSE ({self.pulse_v:.2f} V)", "HIGH")

                t0 = time.perf_counter()
                while not self._stop and (time.perf_counter() - t0) < self.duration_s:
                    time.sleep(0.005)

                self._write_all(task, self.park_v)
                self.signals.status.emit(f"Voltage: PARK ({self.park_v:.2f} V)", "LOW")
                time.sleep(0.05)

        except Exception as e:
            print(f"[ERROR] PulseWorker on {self.channels}: {e}")
            self.signals.status.emit("Error: Check DAQ", "ERROR")
        finally:
            self.signals.done.emit()

class PulseColumn(QWidget):
    def __init__(self, title, channels, update_state_callback, parent=None):
        super().__init__(parent)
        self.channels = channels
        self.update_state_callback = update_state_callback
        self.worker = None

        group = QGroupBox(title)
        outer = QVBoxLayout(group)

        form = QFormLayout()
        self.park_spin = QDoubleSpinBox()
        self.park_spin.setDecimals(3)
        self.park_spin.setRange(-10.0, 10.0)
        self.park_spin.setSingleStep(0.1)
        self.park_spin.setValue(0.0)

        self.pulse_spin = QDoubleSpinBox()
        self.pulse_spin.setDecimals(3)
        self.pulse_spin.setRange(-10.0, 10.0)
        self.pulse_spin.setSingleStep(0.1)
        self.pulse_spin.setValue(10.0)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setDecimals(3)
        self.duration_spin.setRange(0.01, 1000.0)
        self.duration_spin.setSingleStep(1.0)
        self.duration_spin.setValue(0.2)

        form.addRow("Park Volt (V):", self.park_spin)
        form.addRow("Pulse Volt (V):", self.pulse_spin)
        form.addRow("Duration (s):", self.duration_spin)
        outer.addLayout(form)

        self.status_label = QLabel(f"Voltage: PARK (0.0 V)")
        self.status_label.setAlignment(Qt.AlignCenter)
        self._set_style(f"Voltage: PARK (0.0 V)", "LOW")
        outer.addWidget(self.status_label)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet("background-color: green; color: white; font-weight: bold; padding: 6px;")
        self.start_btn.clicked.connect(self.start_pulse)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 6px;")
        self.stop_btn.clicked.connect(self.stop_pulse)

        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        outer.addLayout(btns)
        lay = QVBoxLayout(self)
        lay.addWidget(group)

    def _set_style(self, text, state):
        self.status_label.setText(text)
        if state == "HIGH": self.status_label.setStyleSheet("background-color: #28a745; color: white; font-size: 13px; padding: 6px; border-radius: 4px;")
        elif state == "LOW": self.status_label.setStyleSheet("background-color: #6c757d; color: white; font-size: 13px; padding: 6px; border-radius: 4px;")
        elif state == "ERROR": self.status_label.setStyleSheet("background-color: #dc3545; color: white; font-size: 13px; padding: 6px; border-radius: 4px;")

    def _bind_worker(self, worker):
        worker.signals.status.connect(self._set_style)
        worker.signals.voltage_update.connect(self.update_state_callback)
        worker.signals.done.connect(self._on_done)

    def start_pulse(self):
        if self.worker is not None and self.worker.isRunning(): return
        self.worker = PulseWorker(self.channels, self.park_spin.value(), self.pulse_spin.value(), self.duration_spin.value())
        self._bind_worker(self.worker)
        self.start_btn.setEnabled(False)
        self.worker.start()

    def stop_pulse(self):
        if self.worker is not None and self.worker.isRunning(): self.worker.stop()

    def _on_done(self):
        self.start_btn.setEnabled(True)

    def close(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        super().close()

class VoltageToggleWindow(QWidget):
    def __init__(self, device_name, active_ao_channels, state_update_callback):
        super().__init__()
        self.device_name = device_name
        self.setWindowTitle(f"DAQ Analog Output Pulse Control ({device_name})")

        grid = QGridLayout()
        self.columns = []
        
        if not active_ao_channels:
            lbl = QLabel("No Analog Output channels selected. Go to Channel Config -> Select Active Channels.")
            grid.addWidget(lbl, 0, 0)
        else:
            row, col = 0, 0
            for ao in active_ao_channels:
                idx = ao.replace("AO", "")
                c = PulseColumn(f"Channel: {ao.lower()}", [f"{device_name}/ao{idx}"], state_update_callback)
                self.columns.append(c)
                grid.addWidget(c, row, col)
                col += 1
                if col > 1:
                    col = 0; row += 1
            
            if len(active_ao_channels) > 1:
                all_paths = [f"{device_name}/ao{ao.replace('AO', '')}" for ao in active_ao_channels]
                c_all = PulseColumn(f"Master: ALL ({len(active_ao_channels)} chs)", all_paths, state_update_callback)
                self.columns.append(c_all)
                grid.addWidget(c_all, row, 0, 1, 2)

        root = QVBoxLayout(self)
        root.addLayout(grid)

    def closeEvent(self, event):
        for col in self.columns: col.close()
        event.accept()

# =============================================================================
# MAIN DAQ APPLICATION
# =============================================================================

class ChannelSelectionDialog(QDialog):
    def __init__(self, active_signals, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Active Channels")
        self.setMinimumWidth(800)
        layout = QVBoxLayout(self)
        self.checkboxes = {}
        
        ai_group = QGroupBox("Analog Inputs (AI)")
        ai_layout = QGridLayout()
        r, c = 0, 0
        for sig in ALL_AI_CHANNELS:
            cb = QCheckBox(sig)
            cb.setChecked(sig in active_signals)
            self.checkboxes[sig] = cb
            ai_layout.addWidget(cb, r, c)
            c += 1
            if c > 7:  c = 0; r += 1
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        
        ao_group = QGroupBox("Analog Outputs (AO)")
        ao_layout = QGridLayout()
        r, c = 0, 0
        for sig in ALL_AO_CHANNELS:
            cb = QCheckBox(sig)
            cb.setChecked(sig in active_signals)
            self.checkboxes[sig] = cb
            ao_layout.addWidget(cb, r, c)
            c += 1
        ao_group.setLayout(ao_layout)
        layout.addWidget(ao_group)

        math_group = QGroupBox("Virtual Math Channels")
        math_layout = QGridLayout()
        r, c = 0, 0
        for sig in ALL_MATH_CHANNELS:
            cb = QCheckBox(sig)
            cb.setChecked(sig in active_signals)
            self.checkboxes[sig] = cb
            math_layout.addWidget(cb, r, c)
            c += 1
        math_group.setLayout(math_layout)
        layout.addWidget(math_group)
        
        dmm_group = QGroupBox("External Devices")
        dmm_layout = QVBoxLayout()
        cb = QCheckBox("DMM")
        cb.setChecked("DMM" in active_signals)
        self.checkboxes["DMM"] = cb
        dmm_layout.addWidget(cb)
        dmm_group.setLayout(dmm_layout)
        layout.addWidget(dmm_group)
        
        btns = QHBoxLayout()
        ok_btn = QPushButton("Apply Configuration")
        ok_btn.setStyleSheet("font-weight: bold; background-color: #0078D7; color: white; padding: 5px;")
        ok_btn.clicked.connect(self.accept)
        btns.addStretch()
        btns.addWidget(ok_btn)
        layout.addLayout(btns)

    def get_selected(self):
        selected = []
        for sig in ALL_CHANNELS:
            if sig in self.checkboxes and self.checkboxes[sig].isChecked(): selected.append(sig)
        return selected

class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Export ASCII Data (Interactive)")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(600)
        
        main_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        
        exp_settings = self.parent.export_settings
        
        grid = QGridLayout()
        timestamp = datetime.now().strftime("%H-%M-%S")
        self.filename_input = QLineEdit(f"export_{timestamp}")
        self.resample_rate_input = QLineEdit(exp_settings.get("resample_rate", "10")) 
        self.start_time_input = QLineEdit("0.00")
        self.end_time_input = QLineEdit("Max")
        
        grid.addWidget(QLabel("Export Filename:"), 0, 0)
        grid.addWidget(self.filename_input, 0, 1)
        grid.addWidget(QLabel("Resample Rate (S/s):"), 1, 0)
        grid.addWidget(self.resample_rate_input, 1, 1)
        grid.addWidget(QLabel("Start Time (s):"), 2, 0)
        grid.addWidget(self.start_time_input, 2, 1)
        grid.addWidget(QLabel("End Time (s):"), 3, 0)
        grid.addWidget(self.end_time_input, 3, 1)
        
        left_layout.addLayout(grid)
        
        configs = self.parent.active_channel_configs
        name_map = {cfg["Name"]: cfg["CustomName"] for cfg in configs}
        name_map["DMM"] = "DMM"
        
        ch_group = QGroupBox("Select Channels to Export")
        ch_layout = QGridLayout()
        self.export_ch_cbs = {}
        row, col = 0, 0
        for sig in self.parent.available_signals:
            display_name = f"{name_map.get(sig, sig)} ({sig})"
            if "MATH" in display_name.upper(): 
                continue
            cb = QCheckBox(display_name)
            cb.setChecked(True)
            self.export_ch_cbs[sig] = cb
            ch_layout.addWidget(cb, row, col)
            col += 1
            if col > 2: 
                col = 0; row += 1
        ch_group.setLayout(ch_layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(ch_layout)
        scroll.setWidget(scroll_widget)
        
        ch_group_layout = QVBoxLayout()
        ch_group_layout.addWidget(scroll)
        ch_group.setLayout(ch_group_layout)
        left_layout.addWidget(ch_group)
        
        self.gdrive_cb = QCheckBox("Upload to Google Drive after export")
        self.gdrive_cb.setChecked(exp_settings.get("gdrive_upload", True))
        if not GOOGLE_API_AVAILABLE:
            self.gdrive_cb.setEnabled(False)
            self.gdrive_cb.setText("Google API Missing (pip install google-api-python-client)")
        left_layout.addWidget(self.gdrive_cb)

        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.setStyleSheet("font-weight: bold; background-color: #0078D7; color: white; padding: 10px;")
        self.export_btn.clicked.connect(self.process_export)
        left_layout.addWidget(self.export_btn)
        
        preview_top = QHBoxLayout()
        preview_top.addWidget(QLabel("<b>Preview Channel:</b>"))
        self.preview_cb = QComboBox()
        preview_top.addWidget(self.preview_cb)
        preview_top.addStretch()
        right_layout.addLayout(preview_top)

        # --- PYQTGRAPH REPLACEMENT FOR MATPLOTLIB IN EXPORT ---
        self.preview_plot = pg.PlotWidget(title="Preview Channel")
        self.preview_plot.setLabel('bottom', "Time (s)")
        self.preview_plot.setLabel('left', "Amplitude")
        self.preview_plot.showGrid(x=True, y=True)
        self.preview_curve = self.preview_plot.plot(pen=pg.mkPen('b', width=1.5))
        
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.preview_plot.addItem(self.region, ignoreBounds=True)
        self.region.sigRegionChangeFinished.connect(self.update_region_text)
        right_layout.addWidget(self.preview_plot)
        # ------------------------------------------------------
        
        self.tdms_path = self.parent.current_tdms_filepath
        self.time_data = None
        
        if os.path.exists(self.tdms_path):
            try:
                with TdmsFile.read(self.tdms_path) as tdms_file:
                    group = tdms_file["RawData"]
                    self.time_data = group["Time"][:]
                    self.start_time_input.setText(f"{self.time_data[0]:.3f}")
                    self.end_time_input.setText(f"{self.time_data[-1]:.3f}")
                    self.region.setRegion([self.time_data[0], self.time_data[-1]])
                    
                    for sig in self.parent.available_signals:
                        if sig in group:
                            display_name = f"{name_map.get(sig, sig)} ({sig})"
                            self.preview_cb.addItem(display_name, userData=sig)
                            
                self.preview_cb.currentIndexChanged.connect(self.update_preview_plot)
                self.update_preview_plot() 
            except Exception as e:
                self.preview_plot.setTitle(f"Could not load preview.")
        else:
            self.preview_plot.setTitle("No recording found.")
            self.preview_cb.setEnabled(False)
            
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)

    def update_region_text(self):
        min_x, max_x = self.region.getRegion()
        self.start_time_input.setText(f"{min_x:.3f}")
        self.end_time_input.setText(f"{max_x:.3f}")

    def update_preview_plot(self):
        if self.time_data is None: return
        sig = self.preview_cb.currentData()
        if not sig: return
        
        try:
            with TdmsFile.read(self.tdms_path) as tdms_file:
                group = tdms_file["RawData"]
                if sig in group:
                    y_data = group[sig][:]
                    step = max(1, len(self.time_data) // 3000)
                    self.preview_curve.setData(self.time_data[::step], y_data[::step])
        except Exception: pass
        
    def process_export(self):
        if not os.path.exists(self.tdms_path):
            QMessageBox.critical(self, "Error", "No recorded TDMS file found to export.")
            return
            
        try:
            target_rate = float(self.resample_rate_input.text())
            orig_rate = float(self.parent.read_rate_input.text())
            t_start = float(self.start_time_input.text())
            t_end_text = self.end_time_input.text().strip().lower()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid numeric inputs.")
            return

        self.parent.export_settings["resample_rate"] = self.resample_rate_input.text()
        self.parent.export_settings["gdrive_upload"] = self.gdrive_cb.isChecked()
        self.parent.save_config()

        export_name = f"{self.filename_input.text().strip()}.csv"
        
        date_str = datetime.now().strftime("%d_%m_%Y")
        export_dir = Path(self.parent.output_folder) / date_str / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        out_csv = export_dir / export_name
        
        configs = self.parent.get_current_channel_configs()
        cfg_dict = {c["Name"]: c for c in configs}

        try:
            self.export_btn.setText("Reading TDMS...")
            self.export_btn.setEnabled(False)
            QApplication.processEvents()
            
            with TdmsFile.read(self.tdms_path) as tdms_file:
                group = tdms_file["RawData"]
                time_data = group["Time"][:]
                
                t_end = time_data[-1] if t_end_text == "max" else float(t_end_text)
                mask = (time_data >= t_start) & (time_data <= t_end)
                time_sliced = time_data[mask]
                
                if len(time_sliced) == 0:
                    QMessageBox.warning(self, "Warning", "No data found in range.")
                    self.export_btn.setText("Export to CSV")
                    self.export_btn.setEnabled(True)
                    return
                
                factor = max(1, int(orig_rate / target_rate))
                valid_len = (len(time_sliced) // factor) * factor
                
                self.export_btn.setText("Resampling Data...")
                QApplication.processEvents()
                
                names, units, raw_names, scales, offsets = ["Time"], ["s"], ["Time"], ["1"], ["0"]
                stack = [time_sliced[:valid_len].reshape(-1, factor).mean(axis=1)]
                
                for sig in self.parent.available_signals:
                    if sig in self.export_ch_cbs and self.export_ch_cbs[sig].isChecked():
                        raw_v = group[sig][:][mask]
                        raw_v_sliced = raw_v[:valid_len].reshape(-1, factor).mean(axis=1)
                        
                        if sig in cfg_dict:
                            cfg = cfg_dict[sig]
                            custom_name = cfg["CustomName"]
                            unit = cfg["Unit"]
                            scale = cfg.get("Scale", 1.0)
                            offset = cfg.get("Offset", 0.0)
                            val = (raw_v_sliced * scale) - offset
                        else:
                            custom_name, unit, scale, offset = "DMM", "V", 1.0, 0.0
                            val = raw_v_sliced
                            
                        stack.append(val)
                        names.append(custom_name)
                        units.append(unit)
                        raw_names.append(sig)
                        scales.append(str(scale))
                        offsets.append(str(offset))
                
                arr = np.column_stack(stack)
                self.export_btn.setText("Writing to CSV...")
                QApplication.processEvents()
                
                with open(out_csv, 'w') as f:
                    f.write(",".join(names) + "\n")
                    f.write(",".join(units) + "\n")
                    f.write(",".join(raw_names) + "\n")
                    f.write(",".join(scales) + "\n")
                    f.write(",".join(offsets) + "\n\n") 
                    np.savetxt(f, arr, delimiter=",", fmt="%.6g")

            if self.gdrive_cb.isChecked():
                link = self.parent.gdrive_link_input.text().strip()
                auth_method = self.parent.gdrive_auth_cb.currentText()
                if link:
                    self.parent.start_gdrive_upload(str(out_csv), link, auth_method, is_export=True)

            QMessageBox.information(self, "Success", f"Data exported to:\n{out_csv}")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")
            traceback.print_exc()
            self.export_btn.setText("Export to CSV")
            self.export_btn.setEnabled(True)

class NumericalIndicatorWidget(QWidget):
    def __init__(self, mapping, remove_callback):
        super().__init__()
        self.remove_callback = remove_callback
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        top_layout = QHBoxLayout()
        self.signal_cb = QComboBox()
        for raw_name, custom_name in mapping:
            self.signal_cb.addItem(custom_name, userData=raw_name)
        remove_btn = QPushButton("X")
        remove_btn.setFixedWidth(30)
        remove_btn.clicked.connect(lambda: self.remove_callback(self))
        top_layout.addWidget(self.signal_cb)
        top_layout.addWidget(remove_btn)
        
        self.type_cb = QComboBox()
        self.type_cb.addItems(["Current (100ms avg)", "RMS", "Peak-to-Peak", "Frequency"])
        
        self.name_label = QLabel("")
        self.name_label.setStyleSheet("font-weight: bold; color: gray;")
        self.value_label = QLabel("0.00")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #0055a4;")
        
        layout.addLayout(top_layout)
        layout.addWidget(self.type_cb)
        layout.addWidget(self.name_label)
        layout.addWidget(self.value_label)
        self.setStyleSheet("NumericalIndicatorWidget { border: 2px solid darkgray; border-radius: 6px; background-color: #f8f9fa;}")
        
    def update_display(self, value, unit, raw_name):
        self.name_label.setText(raw_name)
        calc_type = self.type_cb.currentText()
        if calc_type == "Frequency": self.value_label.setText(f"{value:.2f} Hz")
        else: self.value_label.setText(f"{value:.4g} {unit}")

class SubplotSettingsDialog(QDialog):
    def __init__(self, mapping, selected_signals, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Channels for Subplot")
        layout = QVBoxLayout(self)
        self.checkboxes = {}
        grid = QGridLayout()
        row, col = 0, 0
        for raw_sig, display_text in mapping:
            cb = QCheckBox(display_text)
            if raw_sig in selected_signals: cb.setChecked(True)
            self.checkboxes[raw_sig] = cb
            grid.addWidget(cb, row, col)
            col += 1
            if col > 3:  
                col = 0; row += 1
        layout.addLayout(grid)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)
        
    def get_selected_signals(self):
        return [raw_sig for raw_sig, cb in self.checkboxes.items() if cb.isChecked()]

class SubplotConfigWidget(QWidget):
    def __init__(self, index, mapping, remove_callback, config_changed_callback):
        super().__init__()
        self.remove_callback = remove_callback
        self.config_changed_callback = config_changed_callback
        self.mapping = mapping
        self.selected_signals = []
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        top_layout = QHBoxLayout()
        self.title_label = QLabel(f"Subplot {index + 1}")
        self.title_label.setStyleSheet("font-weight: bold;")
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self.open_settings)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.on_remove)
        
        top_layout.addWidget(self.title_label)
        top_layout.addStretch()
        top_layout.addWidget(settings_btn)
        top_layout.addWidget(remove_btn)
        layout.addLayout(top_layout)
        self.setStyleSheet("SubplotConfigWidget { border: 1px solid gray; border-radius: 5px; }")
        
    def open_settings(self):
        dialog = SubplotSettingsDialog(self.mapping, self.selected_signals, self)
        if dialog.exec_():
            self.selected_signals = dialog.get_selected_signals()
            self.config_changed_callback()
            
    def update_index(self, index): self.title_label.setText(f"Subplot {index + 1}")
    def update_mapping(self, mapping): self.mapping = mapping
    def on_remove(self): self.remove_callback(self)
    def get_selected_signals(self): return self.selected_signals
    def set_selected_signals(self, signals): self.selected_signals = signals

class DAQControlApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAQ Control GUI")

        self.DMMread_thread = None
        self.tdms_writer_thread = None
        self.upload_thread = None
        
        # --- MULTIPROCESSING COMPONENTS ---
        self.daq_process = None
        self.math_process = None
        
        self.manager = mp.Manager()
        self.mp_stop_flag = mp.Event()
        
        self.tdms_queue = mp.Queue(maxsize=1000) 
        self.process_queue = mp.Queue(maxsize=100)
        self.gui_update_queue = mp.Queue(maxsize=100)
        
        self.ao_state_dict = self.manager.dict()
        for sig in ALL_AO_CHANNELS: self.ao_state_dict[sig] = 0.0
        self.dmm_buffer_list = self.manager.list()
        
        self.write_stop_flag = threading.Event()
        self.write_task = None
        self.write_task_lock = threading.Lock()
        self.history_lock = threading.Lock() 

        self.start_timestamp = None
        self.sample_nr = 0
        self.output_folder = r"\data"
        self.current_tdms_filepath = ""
        self.export_settings = {}
        
        self.available_signals = ["AI0", "AI1", "AI2", "AI3", "AI4", "AI5", "AO0", "AO1", "AO2", "AO3", "DMM"]
        self.master_channel_configs = {sig: self.get_default_channel_config(sig) for sig in ALL_CHANNELS}
        self.active_channel_configs = [] 
        
        self.history_maxlen = 50000 
        self.history_time = collections.deque(maxlen=self.history_maxlen)
        self.history_data = {sig: collections.deque(maxlen=self.history_maxlen) for sig in ALL_CHANNELS}
        
        self.needs_plot_rebuild = True
        self.latest_math_values = {}
        self.active_math_signals = set()

        self.indicator_widgets = []
        self.fps_queue = collections.deque(maxlen=20)
        self.last_update_time = time.perf_counter()

        self.gui_timer = QTimer()
        self.gui_timer.timeout.connect(self.update_gui)
        self.ao_window = None

        self.tabs = QTabWidget()
        self.main_tab = QWidget()
        self.config_tab = QWidget()
        self.ao_do_tab = QWidget()
        
        self.tabs.addTab(self.main_tab, "Main Control")
        self.tabs.addTab(self.config_tab, "Channel Config")
        self.tabs.addTab(self.ao_do_tab, "Analog and Digital output")
        
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(self.tabs)
        self.setLayout(outer_layout)

        self.setup_ao_do_tab()
        self.setup_config_tab()
        self.setup_main_tab()
        
        self.load_config() 

        try:
            dev_name = self.device_cb.currentData()
            if dev_name:
                nidaqmx.system.Device(dev_name).reset_device()
                do_init = nidaqmx.Task()
                do_init.do_channels.add_do_chan(self.get_do_channel(), line_grouping=LineGrouping.CHAN_PER_LINE)
                do_init.write([False, False])
                do_init.close()
        except Exception as e:
            print(f"[WARN] DAQ Hardware not found at startup: {e}. Check 'Simulate Mode'.")

    def get_default_channel_config(self, raw_name):
        term = "RSE" if raw_name.startswith("AI") and int(raw_name[2:]) >= 16 else "DIFF"
        return {
            "custom_name": raw_name, "term": term, "range": "-10 to 10",
            "sensor": "None", "scale": "1.0", "unit": "V", "offset": "0.0",
            "lpf_on": False, "lpf_cutoff": "10.0", "lpf_order": "4",
            "expression": "AI0 - AI1"
        }

    def get_write_channel(self): return f"{self.device_cb.currentData()}/ao0"
    def get_do_channel(self): return f"{self.device_cb.currentData()}/port0/line0:1"

    def update_ao_state_from_pulse(self, channels, volts):
        for ch in channels:
            raw_sig = ch.split('/')[-1].upper() 
            if raw_sig in self.ao_state_dict:
                self.ao_state_dict[raw_sig] = float(volts)

    def launch_analog_out(self):
        device_name = self.device_cb.currentData()
        active_aos = [s for s in self.available_signals if s.startswith("AO")]
        if self.ao_window is not None: self.ao_window.close()
        self.ao_window = VoltageToggleWindow(device_name, active_aos, self.update_ao_state_from_pulse)
        self.ao_window.show()
        self.ao_window.raise_()

    def update_upload_status(self, msg, color):
        self.upload_status_label.setText(f"Cloud: {msg}")
        self.upload_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def start_gdrive_upload(self, file_path, link, auth_method, is_export=False):
        self.upload_thread = UploadThread(file_path, link, auth_method, is_export)
        self.upload_thread.status_signal.connect(self.update_upload_status)
        self.upload_thread.start()

    def setup_ao_do_tab(self):
        layout = QVBoxLayout(self.ao_do_tab)
        grid = QGridLayout()
        
        self.write_rate_input = QLineEdit("1000")
        self.threshold_input = QLineEdit("0.0002")
        self.write_active_label = QLabel("Write Active (Ramp Profile)")
        self.write_active_label.setStyleSheet("color: grey; font-weight: bold;")
        self.shutdown_label = QLabel("Status: OK")
        self.shutdown_label.setStyleSheet("color: green; font-weight: bold;")
        
        grid.addWidget(QLabel("Write Rate (Hz):"), 0, 0)
        grid.addWidget(self.write_rate_input, 0, 1)
        grid.addWidget(QLabel("Voltage Threshold (V):"), 1, 0)
        grid.addWidget(self.threshold_input, 1, 1)
        grid.addWidget(self.write_active_label, 2, 0, 1, 2)
        grid.addWidget(self.shutdown_label, 3, 0, 1, 2)
        layout.addLayout(grid)
        
        btn_layout = QHBoxLayout()
        self.start_write_btn = QPushButton("Start Ramp Write")
        self.stop_write_btn = QPushButton("Stop Ramp Write")
        self.start_write_btn.clicked.connect(self.start_write)
        self.stop_write_btn.clicked.connect(self.stop_write)
        btn_layout.addWidget(self.start_write_btn)
        btn_layout.addWidget(self.stop_write_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        note = QLabel("\nNote: The above controls are for continuous waveform/ramp output on ao0.\nFor DC Pulse Control, use the 'Open AnalogOut Control' button on the Main tab.")
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)
        layout.addStretch()

    def setup_main_tab(self):
        main_layout = QHBoxLayout(self.main_tab)
        left_panel = QVBoxLayout()
        center_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        left_panel.addWidget(QLabel("<b>Numerical Indicators</b>"))
        self.indicator_scroll_area = QScrollArea()
        self.indicator_scroll_widget = QWidget()
        self.indicator_scroll_layout = QVBoxLayout(self.indicator_scroll_widget)
        self.indicator_scroll_layout.addStretch()
        self.indicator_scroll_area.setWidget(self.indicator_scroll_widget)
        self.indicator_scroll_area.setWidgetResizable(True)
        self.indicator_scroll_area.setMinimumWidth(220)
        self.add_indicator_btn = QPushButton("Add Indicator")
        self.add_indicator_btn.clicked.connect(lambda: self.add_indicator("AI0", "RMS"))
        left_panel.addWidget(self.indicator_scroll_area)
        left_panel.addWidget(self.add_indicator_btn)

        controls_layout = QGridLayout()
        self.read_rate_input = QLineEdit("10000")
        self.average_samples_input = QLineEdit("100")
        self.plot_window_input = QLineEdit("10")
        self.simulate_checkbox = QCheckBox("Simulate Mode")

        self.fps_label = QLabel("GUI FPS: 0.0")
        self.fps_label.setStyleSheet("color: gray; font-weight: bold;")
        
        self.upload_status_label = QLabel("Cloud: Idle")
        self.upload_status_label.setStyleSheet("color: gray; font-weight: bold;")

        self.folder_display = QLabel()
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.folder_display.setText(f"Output Folder: {self.output_folder[-40:]}")
        
        choose_folder_btn = QPushButton("Choose Output Folder")
        choose_folder_btn.clicked.connect(self.select_output_folder)
        self.export_ascii_btn = QPushButton("Export ASCII Data")
        self.export_ascii_btn.setStyleSheet("font-weight: bold; background-color: #0078D7; color: white;")
        self.export_ascii_btn.clicked.connect(self.open_export_dialog)
        self.open_ao_btn = QPushButton("Open AnalogOut Control Pop-up")
        self.open_ao_btn.setStyleSheet("font-weight: bold; background-color: #ff9900; color: black;")
        self.open_ao_btn.clicked.connect(self.launch_analog_out)
        
        controls_layout.addWidget(QLabel("Read Rate (Hz):"), 0, 0)
        controls_layout.addWidget(self.read_rate_input, 0, 1)
        controls_layout.addWidget(QLabel("Samples to Average Over:"), 1, 0)
        controls_layout.addWidget(self.average_samples_input, 1, 1)
        controls_layout.addWidget(QLabel("Plot View Window (s, 0=All):"), 2, 0)
        controls_layout.addWidget(self.plot_window_input, 2, 1)
        
        controls_layout.addWidget(self.folder_display, 3, 0, 1, 2)
        controls_layout.addWidget(choose_folder_btn, 4, 0, 1, 2)
        controls_layout.addWidget(self.export_ascii_btn, 5, 0, 1, 2)
        controls_layout.addWidget(self.open_ao_btn, 6, 0, 1, 2)
        
        status_row = QHBoxLayout()
        status_row.addWidget(self.simulate_checkbox)
        status_row.addStretch()
        status_row.addWidget(self.upload_status_label)
        status_row.addStretch()
        status_row.addWidget(self.fps_label)
        controls_layout.addLayout(status_row, 7, 0, 1, 2)
        
        plot_control_layout = QVBoxLayout()
        plot_control_layout.addWidget(QLabel("<b>Dynamic Plot Configuration</b>"))
        self.plot_scroll_area = QScrollArea()
        self.plot_scroll_widget = QWidget()
        self.plot_scroll_layout = QVBoxLayout(self.plot_scroll_widget)
        self.plot_scroll_layout.addStretch()
        self.plot_scroll_area.setWidget(self.plot_scroll_widget)
        self.plot_scroll_area.setWidgetResizable(True)
        self.add_subplot_btn = QPushButton("Add New Subplot")
        self.add_subplot_btn.clicked.connect(lambda: self.add_subplot())
        plot_control_layout.addWidget(self.plot_scroll_area)
        plot_control_layout.addWidget(self.add_subplot_btn)

        self.start_read_btn = QPushButton("Start Read")
        self.stop_read_btn = QPushButton("Stop Read")
        self.exit_btn = QPushButton("Exit")
        self.reset_btn = QPushButton("Reset All")

        self.start_read_btn.clicked.connect(self.start_read)
        self.stop_read_btn.clicked.connect(self.stop_read)
        self.exit_btn.clicked.connect(self.exit_application)
        self.reset_btn.clicked.connect(self.reset_all)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_read_btn)
        btn_layout.addWidget(self.stop_read_btn)
        btn_layout.addWidget(self.exit_btn)
        btn_layout.addWidget(self.reset_btn)

# --- PYQTGRAPH REPLACEMENT FOR MATPLOTLIB ---
        self.graph_layout = pg.GraphicsLayoutWidget()
        self.plots = []
        self.curves = {}
        self.subplot_widgets = []  # <--- ADD THIS LINE BACK

        right_panel.addLayout(controls_layout)
        right_panel.addLayout(plot_control_layout) 
        center_panel.addWidget(self.graph_layout)
        center_panel.addLayout(btn_layout)

        main_layout.addLayout(left_panel, stretch=1)
        main_layout.addLayout(center_panel, stretch=4)
        main_layout.addLayout(right_panel, stretch=1)

    def refresh_devices(self):
        current_data = self.device_cb.currentData()
        self.device_cb.clear()
        devs = []
        try:
            sys_local = nidaqmx.system.System.local()
            for d in sys_local.devices: devs.append((d.name, d.product_type))
            if not devs: devs = [("Dev1", "Simulated Device")]
        except Exception:
            devs = [("Dev1", "Simulated Device")]
            
        idx_to_set = 0
        for i, (name, ptype) in enumerate(devs):
            self.device_cb.addItem(f"{name} ({ptype})", userData=name)
            if name == current_data: idx_to_set = i
        
        if self.device_cb.count() > 0: self.device_cb.setCurrentIndex(idx_to_set)

    def update_device_labels(self, _):
        new_device = self.device_cb.currentData()
        if not new_device: return
        for ch in self.channel_ui_configs:
            raw_name = ch['name']
            if not raw_name.startswith("MATH"):
                ch['ch_label'].setText(f"{new_device}/{raw_name.lower()} ({raw_name})")

    def open_channel_selector(self):
        self.cache_current_ui_configs()
        dialog = ChannelSelectionDialog(self.available_signals, self)
        if dialog.exec_():
            self.available_signals = dialog.get_selected()
            self.rebuild_config_tab()
            self.apply_config_update()

    def cache_current_ui_configs(self):
        for ch in self.channel_ui_configs:
            raw_name = ch['name']
            if raw_name.startswith("MATH"):
                self.master_channel_configs[raw_name].update({
                    "custom_name": ch["custom_name_input"].text(),
                    "expression": ch["expr_input"].text(),
                    "unit": ch["unit_input"].text()
                })
            else:
                self.master_channel_configs[raw_name].update({
                    "custom_name": ch["custom_name_input"].text(),
                    "term": ch["term_cb"].currentText(),
                    "range": ch["range_cb"].currentText(),
                    "sensor": ch["sensor_cb"].currentText(),
                    "scale": ch["scale_input"].text(),
                    "unit": ch["unit_input"].text(),
                    "offset": ch["offset_input"].text()
                })
                if 'lpf_cb' in ch:
                    self.master_channel_configs[raw_name].update({
                        "lpf_on": ch["lpf_cb"].isChecked(),
                        "lpf_cutoff": ch["lpf_cut"].text(),
                        "lpf_order": ch["lpf_ord"].currentText()
                    })

    def apply_batch_config(self):
        b_term = self.batch_term.currentText()
        b_range = self.batch_range.currentText()
        b_sensor = self.batch_sensor.currentText()
        for ch_ui in self.channel_ui_configs:
            if ch_ui['name'].startswith("AI"):
                idx = int(ch_ui['name'][2:])
                ch_ui['range_cb'].setCurrentText(b_range)
                ch_ui['sensor_cb'].setCurrentText(b_sensor)
                if idx >= 16 and b_term == "DIFF": ch_ui['term_cb'].setCurrentText("RSE")
                else: ch_ui['term_cb'].setCurrentText(b_term)

    def calibrate_single_offset(self, raw_name, scale_input, offset_input, term_cb, range_cb, sensor_cb):
        if self.simulate_checkbox.isChecked():
            offset_input.setText("0.000")
            return
        try: scale_val = float(scale_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid Scale Factor.")
            return
        try:
            with nidaqmx.Task() as task:
                dev_prefix = self.device_cb.currentData()
                term_path = f"{dev_prefix}/{raw_name.lower()}"
                if sensor_cb.currentText() == "Type K":
                    task.ai_channels.add_ai_thrmcpl_chan(term_path, thermocouple_type=ThermocoupleType.K, cjc_source=CJCSource.BUILT_IN)
                else:
                    config_map = {"RSE": TerminalConfiguration.RSE, "NRSE": TerminalConfiguration.NRSE, "DIFF": TerminalConfiguration.DIFF}
                    r_str = range_cb.currentText().split(" to ")
                    task.ai_channels.add_ai_voltage_chan(term_path, terminal_config=config_map[term_cb.currentText()], min_val=float(r_str[0]), max_val=float(r_str[1]))
                    
                task.timing.cfg_samp_clk_timing(rate=1000, sample_mode=AcquisitionType.FINITE, samps_per_chan=1000)
                raw_mean = np.mean(task.read(number_of_samples_per_channel=1000, timeout=3.0))
                offset_input.setText(f"{(raw_mean * scale_val):.6g}")
        except Exception as e: QMessageBox.critical(self, "Hardware Error", f"Failed to measure offset:\n{e}")

    def setup_config_tab(self):
        main_lay = QVBoxLayout(self.config_tab)
        
        dev_layout = QHBoxLayout()
        dev_layout.addWidget(QLabel("<b>DAQ Device:</b>"))
        self.device_cb = QComboBox()
        self.refresh_devices()
        self.device_cb.currentIndexChanged.connect(self.update_device_labels)
        dev_layout.addWidget(self.device_cb)
        self.refresh_dev_btn = QPushButton("Refresh Devices")
        self.refresh_dev_btn.clicked.connect(self.refresh_devices)
        dev_layout.addWidget(self.refresh_dev_btn)
        dev_layout.addSpacing(20)
        self.select_ch_btn = QPushButton("Select Active Channels")
        self.select_ch_btn.setStyleSheet("font-weight: bold; background-color: #17a2b8; color: white; padding: 6px;")
        self.select_ch_btn.clicked.connect(self.open_channel_selector)
        dev_layout.addWidget(self.select_ch_btn)
        dev_layout.addStretch()
        main_lay.addLayout(dev_layout)

        self.batch_group = QGroupBox("Batch Configure Active Analog Inputs")
        batch_lay = QGridLayout()
        batch_lay.addWidget(QLabel("Terminal:"), 0, 0)
        self.batch_term = QComboBox()
        self.batch_term.addItems(["RSE", "NRSE", "DIFF"])
        batch_lay.addWidget(self.batch_term, 0, 1)
        batch_lay.addWidget(QLabel("Voltage Range:"), 0, 2)
        self.batch_range = QComboBox()
        self.batch_range.addItems(["-10 to 10", "-5 to 5", "-2.5 to 2.5", "-0.2 to 0.2"])
        batch_lay.addWidget(self.batch_range, 0, 3)
        batch_lay.addWidget(QLabel("Sensor:"), 0, 4)
        self.batch_sensor = QComboBox()
        self.batch_sensor.addItems(["None", "Type K"])
        batch_lay.addWidget(self.batch_sensor, 0, 5)
        self.batch_apply_btn = QPushButton("Apply to All AI")
        self.batch_apply_btn.setStyleSheet("font-weight: bold; background-color: #6c757d; color: white;")
        self.batch_apply_btn.clicked.connect(self.apply_batch_config)
        batch_lay.addWidget(self.batch_apply_btn, 0, 6)
        self.batch_group.setLayout(batch_lay)
        main_lay.addWidget(self.batch_group)

        self.config_scroll = QScrollArea()
        self.config_scroll.setWidgetResizable(True)
        self.config_widget = QWidget()
        self.config_grid_layout = QVBoxLayout(self.config_widget)
        self.config_grid = QGridLayout()
        self.config_grid_layout.addLayout(self.config_grid)
        self.config_grid_layout.addStretch()
        self.config_scroll.setWidget(self.config_widget)
        main_lay.addWidget(self.config_scroll)
        
        bottom_grid = QGridLayout()
        bottom_grid.addWidget(QLabel("<b>Keithley DMM IP:</b>"), 0, 0)
        self.Keithley_DMM_IP = QLineEdit("169.254.169.37")
        bottom_grid.addWidget(self.Keithley_DMM_IP, 0, 1, 1, 4) 
        
        bottom_grid.addWidget(QLabel("<b>Google Drive Target Folder Link:</b>"), 1, 0)
        self.gdrive_link_input = QLineEdit("")
        self.gdrive_link_input.setPlaceholderText("https://drive.google.com/drive/folders/...")
        bottom_grid.addWidget(self.gdrive_link_input, 1, 1)
        
        bottom_grid.addWidget(QLabel("<b>Auth Method:</b>"), 1, 2)
        self.gdrive_auth_cb = QComboBox()
        self.gdrive_auth_cb.addItems(["OAuth 2.0 (User Login)", "Service Account (Robot)"])
        bottom_grid.addWidget(self.gdrive_auth_cb, 1, 3)
        
        self.gdrive_help_btn = QPushButton("Help")
        self.gdrive_help_btn.setStyleSheet("font-weight: bold; background-color: #6c757d; color: white;")
        self.gdrive_help_btn.clicked.connect(self.show_gdrive_help)
        bottom_grid.addWidget(self.gdrive_help_btn, 1, 4)

        main_lay.addLayout(bottom_grid)
        
        bottom_hlay = QHBoxLayout()
        self.apply_config_btn = QPushButton("Save Settings & Update UI")
        self.apply_config_btn.setStyleSheet("font-weight: bold; background-color: #28a745; color: white; padding: 8px;")
        self.apply_config_btn.clicked.connect(self.apply_config_update)
        bottom_hlay.addStretch()
        bottom_hlay.addWidget(self.apply_config_btn)
        main_lay.addLayout(bottom_hlay)
        self.rebuild_config_tab()

    def show_gdrive_help(self):
        dialog = GDriveHelpDialog(self)
        dialog.exec_()
        
    def show_math_help(self):
        dialog = MathHelpDialog(self)
        dialog.exec_()

    def rebuild_config_tab(self):
        while self.config_grid.count():
            item = self.config_grid.takeAt(0)
            widget = item.widget()
            if widget is not None: widget.deleteLater()

        self.channel_ui_configs = []
        term_options = ["RSE", "NRSE", "DIFF"]
        term_options_high = ["RSE", "NRSE"] 
        range_options = ["-10 to 10", "-5 to 5", "-2.5 to 2.5", "-0.2 to 0.2"]
        sensor_options = ["None", "Type K"]

        def make_sensor_callback(sensor_cb, unit_input):
            def callback(index):
                if sensor_cb.currentText() != "None":
                    unit_input.setText("°C")
            return callback

        headers = ["Channel", "Custom Name", "Terminal Config", "Voltage Range", "Sensor Type", "Scale Factor", "Unit", "Offset", "Zero", "LPF On", "Cutoff", "Order"]
        for col, h in enumerate(headers): self.config_grid.addWidget(QLabel(f"<b>{h}</b>"), 0, col)

        row_idx = 1
        dev_prefix = self.device_cb.currentData()
        import functools
        
        active_ai = [s for s in self.available_signals if s.startswith("AI")]
        if active_ai:
            ai_label = QLabel("<b>Analog Inputs (AI)</b>")
            ai_label.setStyleSheet("color: #0055a4; font-size: 14px; margin-top: 10px; margin-bottom: 5px;")
            self.config_grid.addWidget(ai_label, row_idx, 0, 1, 12)
            row_idx += 1

        for raw_name in self.available_signals:
            if not raw_name.startswith("AI"): continue

            cfg = self.master_channel_configs[raw_name]
            ch_label = QLabel(f"{dev_prefix}/{raw_name.lower()} ({raw_name})")
            custom_name_input = QLineEdit(cfg.get("custom_name", raw_name))
            term_cb = QComboBox()
            if int(raw_name[2:]) >= 16:
                term_cb.addItems(term_options_high)
                if cfg.get("term", "RSE") == "DIFF": term_cb.setCurrentText("RSE")
                else: term_cb.setCurrentText(cfg.get("term", "RSE"))
            else: 
                term_cb.addItems(term_options)
                term_cb.setCurrentText(cfg.get("term", "DIFF"))
                
            range_cb = QComboBox()
            range_cb.addItems(range_options)
            range_cb.setCurrentText(cfg.get("range", "-10 to 10"))
            
            sensor_cb = QComboBox()
            sensor_cb.addItems(sensor_options)
            
            scale_input = QLineEdit(cfg.get("scale", "1.0"))
            unit_input = QLineEdit(cfg.get("unit", "V"))
            offset_input = QLineEdit(cfg.get("offset", "0.0"))

            zero_btn = QPushButton("Zero")
            zero_btn.clicked.connect(functools.partial(self.calibrate_single_offset, raw_name, scale_input, offset_input, term_cb, range_cb, sensor_cb))
            
            lpf_cb = QCheckBox()
            lpf_cb.setChecked(cfg.get("lpf_on", False))
            lpf_cut = QLineEdit(cfg.get("lpf_cutoff", "10.0"))
            lpf_ord = QComboBox()
            lpf_ord.addItems(["2", "4", "6"])
            lpf_ord.setCurrentText(str(cfg.get("lpf_order", "4")))

            sensor_cb.currentIndexChanged.connect(make_sensor_callback(sensor_cb, unit_input))
            sensor_cb.setCurrentText(cfg.get("sensor", "None")) 

            self.config_grid.addWidget(ch_label, row_idx, 0)
            self.config_grid.addWidget(custom_name_input, row_idx, 1)
            self.config_grid.addWidget(term_cb, row_idx, 2)
            self.config_grid.addWidget(range_cb, row_idx, 3)
            self.config_grid.addWidget(sensor_cb, row_idx, 4)
            self.config_grid.addWidget(scale_input, row_idx, 5)
            self.config_grid.addWidget(unit_input, row_idx, 6)
            self.config_grid.addWidget(offset_input, row_idx, 7)
            self.config_grid.addWidget(zero_btn, row_idx, 8)
            self.config_grid.addWidget(lpf_cb, row_idx, 9, alignment=Qt.AlignCenter)
            self.config_grid.addWidget(lpf_cut, row_idx, 10)
            self.config_grid.addWidget(lpf_ord, row_idx, 11)

            self.channel_ui_configs.append({
                "name": raw_name, "ch_label": ch_label, "custom_name_input": custom_name_input,
                "term_cb": term_cb, "range_cb": range_cb, "sensor_cb": sensor_cb,
                "scale_input": scale_input, "unit_input": unit_input, "offset_input": offset_input,
                "lpf_cb": lpf_cb, "lpf_cut": lpf_cut, "lpf_ord": lpf_ord
            })
            row_idx += 1

        active_ao = [s for s in self.available_signals if s.startswith("AO")]
        if active_ao:
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            self.config_grid.addWidget(line, row_idx, 0, 1, 12)
            row_idx += 1
            ao_label = QLabel("<b>Analog Outputs (AO)</b>")
            ao_label.setStyleSheet("color: #0055a4; font-size: 14px; margin-top: 5px; margin-bottom: 5px;")
            self.config_grid.addWidget(ao_label, row_idx, 0, 1, 12)
            row_idx += 1
            
            for raw_name in active_ao:
                cfg = self.master_channel_configs[raw_name]
                ch_label = QLabel(f"{dev_prefix}/{raw_name.lower()} ({raw_name})")
                custom_name_input = QLineEdit(cfg.get("custom_name", raw_name))
                unit_input = QLineEdit("V")
                unit_input.setEnabled(False)
                
                self.config_grid.addWidget(ch_label, row_idx, 0)
                self.config_grid.addWidget(custom_name_input, row_idx, 1)
                self.config_grid.addWidget(unit_input, row_idx, 6)
                self.channel_ui_configs.append({
                    "name": raw_name, "ch_label": ch_label, "custom_name_input": custom_name_input,
                    "term_cb": QComboBox(), "range_cb": QComboBox(), "sensor_cb": QComboBox(),
                    "scale_input": QLineEdit("1.0"), "unit_input": unit_input, "offset_input": QLineEdit("0.0")
                })
                row_idx += 1

        active_math = [s for s in self.available_signals if s.startswith("MATH")]
        if active_math:
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            self.config_grid.addWidget(line, row_idx, 0, 1, 12)
            row_idx += 1
            
            m_lay = QHBoxLayout()
            m_label = QLabel("<b>Virtual Math Channels</b>")
            m_label.setStyleSheet("color: #0055a4; font-size: 14px; margin-top: 5px; margin-bottom: 5px;")
            m_help = QPushButton("Math Help")
            m_help.setStyleSheet("font-weight:bold; background-color:#ffc107;")
            m_help.clicked.connect(self.show_math_help)
            m_lay.addWidget(m_label)
            m_lay.addWidget(m_help)
            m_lay.addStretch()
            self.config_grid.addLayout(m_lay, row_idx, 0, 1, 12)
            row_idx += 1
            
            for raw_name in active_math:
                cfg = self.master_channel_configs[raw_name]
                ch_label = QLabel(f"{raw_name}")
                custom_name_input = QLineEdit(cfg.get("custom_name", raw_name))
                expr_input = QLineEdit(cfg.get("expression", "AI0 - AI1"))
                unit_input = QLineEdit(cfg.get("unit", "Value"))
                
                self.config_grid.addWidget(ch_label, row_idx, 0)
                self.config_grid.addWidget(custom_name_input, row_idx, 1)
                self.config_grid.addWidget(expr_input, row_idx, 2, 1, 4) 
                self.config_grid.addWidget(unit_input, row_idx, 6)
                
                self.channel_ui_configs.append({
                    "name": raw_name, "ch_label": ch_label, "custom_name_input": custom_name_input,
                    "expr_input": expr_input, "unit_input": unit_input
                })
                row_idx += 1

    def apply_config_update(self):
        self.cache_current_ui_configs()
        self.active_channel_configs = self.get_current_channel_configs()
        self.latest_math_values = {sig: {"Current (100ms avg)": 0.0, "RMS": 0.0, "Peak-to-Peak": 0.0, "Frequency": 0.0} for sig in self.available_signals}
        ind_mapping = []
        sub_mapping = []
        for cfg in self.active_channel_configs:
            raw, custom = cfg['Name'], cfg['CustomName']
            ind_mapping.append((raw, custom))
            sub_mapping.append((raw, f"{custom} ({raw})"))
        if "DMM" in self.available_signals:
            ind_mapping.append(("DMM", "DMM"))
            sub_mapping.append(("DMM", "DMM (DMM)"))
        for sub in self.subplot_widgets: sub.update_mapping(sub_mapping)
        for ind in self.indicator_widgets:
            current_raw = ind.signal_cb.currentData()
            ind.signal_cb.clear()
            for raw, custom in ind_mapping: ind.signal_cb.addItem(custom, userData=raw)
            idx = ind.signal_cb.findData(current_raw)
            if idx >= 0: ind.signal_cb.setCurrentIndex(idx)
            elif ind.signal_cb.count() > 0: ind.signal_cb.setCurrentIndex(0)
        self.save_config()
        self.needs_plot_rebuild = True

    def flag_plot_rebuild(self): self.needs_plot_rebuild = True

    def add_indicator(self, default_signal="AI0", default_type="Current (100ms avg)"):
        ind_mapping = []
        for cfg in self.active_channel_configs: ind_mapping.append((cfg['Name'], cfg['CustomName']))
        if "DMM" in self.available_signals: ind_mapping.append(("DMM", "DMM"))
        widget = NumericalIndicatorWidget(ind_mapping, self.remove_indicator)
        idx = widget.signal_cb.findData(default_signal)
        if idx >= 0: widget.signal_cb.setCurrentIndex(idx)
        widget.type_cb.setCurrentText(default_type)
        self.indicator_scroll_layout.insertWidget(len(self.indicator_widgets), widget)
        self.indicator_widgets.append(widget)

    def remove_indicator(self, widget):
        self.indicator_scroll_layout.removeWidget(widget)
        widget.deleteLater()
        self.indicator_widgets.remove(widget)

    def add_subplot(self, default_signals=None):
        idx = len(self.subplot_widgets)
        sub_mapping = []
        for cfg in self.active_channel_configs: sub_mapping.append((cfg['Name'], f"{cfg['CustomName']} ({cfg['Name']})"))
        if "DMM" in self.available_signals: sub_mapping.append(("DMM", "DMM (DMM)"))
        widget = SubplotConfigWidget(idx, sub_mapping, self.remove_subplot, self.flag_plot_rebuild)
        if default_signals: widget.set_selected_signals(default_signals)
        self.plot_scroll_layout.insertWidget(len(self.subplot_widgets), widget)
        self.subplot_widgets.append(widget)
        self.needs_plot_rebuild = True

    def remove_subplot(self, widget):
        self.plot_scroll_layout.removeWidget(widget)
        widget.deleteLater()
        self.subplot_widgets.remove(widget)
        for i, w in enumerate(self.subplot_widgets): w.update_index(i)
        self.needs_plot_rebuild = True

    def save_config(self):
        self.cache_current_ui_configs()
        config = {
            "main": {
                "device": self.device_cb.currentData(),
                "write_rate": self.write_rate_input.text(),
                "read_rate": self.read_rate_input.text(),
                "avg_samples": self.average_samples_input.text(),
                "plot_window": self.plot_window_input.text(),
                "threshold": self.threshold_input.text(),
                "dmm_ip": self.Keithley_DMM_IP.text(),
                "gdrive_link": self.gdrive_link_input.text(),
                "gdrive_auth": self.gdrive_auth_cb.currentText(),
                "simulate": self.simulate_checkbox.isChecked(),
                "output_folder": self.output_folder,
                "available_signals": self.available_signals
            },
            "master_channels": self.master_channel_configs,
            "export_settings": self.export_settings,
            "subplots": [sub.get_selected_signals() for sub in self.subplot_widgets],
            "indicators": [{"signal": ind.signal_cb.currentData(), "type": ind.type_cb.currentText()} for ind in self.indicator_widgets]
        }
        try:
            with open("daq_config.json", "w") as f: json.dump(config, f, indent=4)
        except Exception: pass

    def load_config(self):
        if not os.path.exists("daq_config.json"): 
            self.add_subplot(["AI0", "AI1"]); self.add_subplot(["AI2", "AI3"]); self.add_indicator("AI0", "RMS")
            return False
            
        try:
            with open("daq_config.json", "r") as f: config = json.load(f)
            main_cfg = config.get("main", {})
            if "device" in main_cfg:
                idx = self.device_cb.findData(main_cfg["device"])
                if idx >= 0: self.device_cb.setCurrentIndex(idx)
            if "write_rate" in main_cfg: self.write_rate_input.setText(main_cfg["write_rate"])
            if "read_rate" in main_cfg: self.read_rate_input.setText(main_cfg["read_rate"])
            if "avg_samples" in main_cfg: self.average_samples_input.setText(main_cfg["avg_samples"])
            if "plot_window" in main_cfg: self.plot_window_input.setText(main_cfg["plot_window"])
            if "threshold" in main_cfg: self.threshold_input.setText(main_cfg["threshold"])
            if "dmm_ip" in main_cfg: self.Keithley_DMM_IP.setText(main_cfg["dmm_ip"])
            if "gdrive_link" in main_cfg: self.gdrive_link_input.setText(main_cfg["gdrive_link"])
            
            if "gdrive_auth" in main_cfg: 
                idx = self.gdrive_auth_cb.findText(main_cfg["gdrive_auth"])
                if idx >= 0: self.gdrive_auth_cb.setCurrentIndex(idx)

            if "simulate" in main_cfg: self.simulate_checkbox.setChecked(main_cfg["simulate"])
            if "output_folder" in main_cfg: 
                self.output_folder = main_cfg["output_folder"]
                self.folder_display.setText(f"Output Folder: {self.output_folder[-40:]}")

            if "available_signals" in main_cfg: self.available_signals = main_cfg["available_signals"]
            if "master_channels" in config:
                for k, v in config["master_channels"].items():
                    if k in self.master_channel_configs: self.master_channel_configs[k].update(v)
            
            self.export_settings = config.get("export_settings", {})

            self.rebuild_config_tab()
            self.apply_config_update()

            while self.subplot_widgets: self.remove_subplot(self.subplot_widgets[0])
            while self.indicator_widgets: self.remove_indicator(self.indicator_widgets[0])

            for sub_signals in config.get("subplots", []): self.add_subplot(sub_signals)
            for ind_cfg in config.get("indicators", []): self.add_indicator(ind_cfg.get("signal", "AI0"), ind_cfg.get("type", "RMS"))
            return True
        except Exception: return False

    def get_current_channel_configs(self):
        config_map = {
            "RSE": TerminalConfiguration.RSE, "NRSE": TerminalConfiguration.NRSE,
            "DIFF": TerminalConfiguration.DIFF, "PSEUDO_DIFF": TerminalConfiguration.PSEUDO_DIFF
        }
        def parse_range(r_str):
            try: return (float(r_str.split(" to ")[0]), float(r_str.split(" to ")[1]))
            except: return (-10.0, 10.0)

        dev_prefix = self.device_cb.currentData()
        daq_configs = []
        for ch in self.channel_ui_configs:
            if ch['name'].startswith("MATH"):
                daq_configs.append({
                    'Name': ch['name'],
                    'CustomName': ch['custom_name_input'].text().strip() or ch['name'],
                    'Expression': ch['expr_input'].text().strip(),
                    'Unit': ch['unit_input'].text().strip(),
                    'Scale': 1.0, 'Offset': 0.0 
                })
                continue

            sensor_type = ch['sensor_cb'].currentText() if 'sensor_cb' in ch else "None"
            try: scale_val = float(ch['scale_input'].text())
            except ValueError: scale_val = 1.0 
            try: offset_val = float(ch['offset_input'].text())
            except ValueError: offset_val = 0.0

            lpf_on = ch['lpf_cb'].isChecked() if 'lpf_cb' in ch else False
            try: lpf_cut = float(ch['lpf_cut'].text()) if 'lpf_cut' in ch else 10.0
            except ValueError: lpf_cut = 10.0
            lpf_ord = int(ch['lpf_ord'].currentText()) if 'lpf_ord' in ch else 4

            t_cb_text = ch['term_cb'].currentText() if 'term_cb' in ch and ch['term_cb'].count() > 0 else "RSE"
            r_cb_text = ch['range_cb'].currentText() if 'range_cb' in ch and ch['range_cb'].count() > 0 else "-10 to 10"

            daq_configs.append({
                'Name': ch['name'], 'Terminal': f"{dev_prefix}/{ch['name'].lower()}",
                'CustomName': ch['custom_name_input'].text().strip() or ch['name'],
                'Config': config_map.get(t_cb_text, TerminalConfiguration.RSE),
                'Range': parse_range(r_cb_text),
                'SensorType': sensor_type, 'Scale': scale_val,
                'Unit': ch['unit_input'].text().strip(), 'Offset': offset_val,
                'LPF_On': lpf_on, 'LPF_Cutoff': lpf_cut, 'LPF_Order': lpf_ord
            })
        return daq_configs

    def start_read(self):
        self.apply_config_update()
        if self.DMMread_thread and self.DMMread_thread.is_alive(): return   
        if self.daq_process and self.daq_process.is_alive(): return     
        if self.math_process and self.math_process.is_alive(): return
        if self.tdms_writer_thread and self.tdms_writer_thread.is_alive(): return
        
        self.mp_stop_flag.clear()
        if self.start_timestamp is None: self.start_timestamp = time.time()
            
        with self.history_lock:
            self.history_time.clear()
            for sig in self.available_signals: self.history_data[sig].clear()
                
        self.needs_plot_rebuild = True
        self.active_math_signals = set(ind.signal_cb.currentData() for ind in self.indicator_widgets)

        self.read_rate_input.setEnabled(False)
        self.average_samples_input.setEnabled(False)

        while not self.tdms_queue.empty(): self.tdms_queue.get()
        while not self.process_queue.empty(): self.process_queue.get()
        while not self.gui_update_queue.empty(): self.gui_update_queue.get()
        del self.dmm_buffer_list[:]

        # Thread Setup
        self.tdms_writer_thread = threading.Thread(target=self.tdms_writer_thread_func)
        self.DMMread_thread = threading.Thread(target=self.DMM_read)
        self.tdms_writer_thread.start()
        self.DMMread_thread.start()
        
        # Multiprocessing setup 
        configs = self.active_channel_configs
        cfg_dict = {c["Name"]: c for c in configs}
        active_ai_configs = [c for c in configs if c['Name'].startswith("AI")]
        hw_signals = [sig for sig in self.available_signals if not sig.startswith("MATH")]
        math_signals = [sig for sig in self.available_signals if sig.startswith("MATH")]
        
        try: read_rate = float(self.read_rate_input.text())
        except ValueError: read_rate = 10000.0
        try: average_samples = max(1, int(self.average_samples_input.text()))
        except ValueError: average_samples = 100
        samples_per_read = max(1, int(read_rate // 10)) 
        
        simulate = self.simulate_checkbox.isChecked()
        has_dmm = "DMM" in self.available_signals
        n_ai = len(active_ai_configs)
        n_ao = sum(1 for c in configs if c['Name'].startswith("AO"))
        
        self.daq_process = mp.Process(target=daq_read_worker, args=(
            self.mp_stop_flag, simulate, read_rate, samples_per_read, active_ai_configs, 
            n_ai, n_ao, has_dmm, self.available_signals, self.ao_state_dict, self.dmm_buffer_list, 
            self.tdms_queue, self.process_queue))
            
        self.math_process = mp.Process(target=math_processing_worker, args=(
            self.mp_stop_flag, read_rate, average_samples, self.available_signals, 
            hw_signals, math_signals, cfg_dict, self.process_queue, self.gui_update_queue))

        time.sleep(0.1)
        self.daq_process.start()
        self.math_process.start()
        
        self.last_update_time = time.perf_counter()
        self.gui_timer.start(100)

    def stop_read(self):
        self.mp_stop_flag.set()
        self.gui_timer.stop()
        
        self.read_rate_input.setEnabled(True)
        self.average_samples_input.setEnabled(True)
        
        threading.Thread(target=self._wait_and_upload).start()

    def _wait_and_upload(self):
        if self.tdms_writer_thread and self.tdms_writer_thread.is_alive():
            self.tdms_writer_thread.join(timeout=5.0)
            
        link = self.gdrive_link_input.text().strip()
        auth_method = self.gdrive_auth_cb.currentText()
        if link and os.path.exists(self.current_tdms_filepath):
            self.start_gdrive_upload(self.current_tdms_filepath, link, auth_method, is_export=False)

    def tdms_writer_thread_func(self):
        filename = self.generate_filename("raw_data")
        self.current_tdms_filepath = str(filename)
        active_ai = [s for s in self.available_signals if s.startswith("AI")]
        active_ao = [s for s in self.available_signals if s.startswith("AO")]
        has_dmm = "DMM" in self.available_signals

        try:
            with TdmsWriter(str(filename)) as writer:
                while not self.mp_stop_flag.is_set() or not self.tdms_queue.empty():
                    try: chunk = self.tdms_queue.get(timeout=0.2)
                    except queue.Empty: continue
                    time_arr, ai_data, ao_data, dmm_data = chunk
                    channels = [ChannelObject("RawData", "Time", time_arr)]
                    for i, sig in enumerate(active_ai): channels.append(ChannelObject("RawData", sig, ai_data[i]))
                    for i, sig in enumerate(active_ao): channels.append(ChannelObject("RawData", sig, ao_data[i]))
                    if has_dmm: channels.append(ChannelObject("RawData", "DMM", dmm_data.flatten()))
                    writer.write_segment(channels)
        except Exception: pass

    def DMM_read(self):
        if "DMM" not in self.available_signals: return
        if self.simulate_checkbox.isChecked():
            while not self.mp_stop_flag.is_set():
                self.dmm_buffer_list.append(float(np.random.uniform(-0.1, 0.1)))
                time.sleep(0.1)
            return
        inst = None
        try:
            inst = DMM6510readout.write_script_to_Keithley(self.Keithley_DMM_IP.text(), "0.05")
            while not self.mp_stop_flag.is_set():
                self.dmm_buffer_list.append(float(DMM6510readout.read_data(inst)))
        except Exception: pass
        finally:
            if inst is not None: 
                try: DMM6510readout.stop_instrument(inst)
                except: pass

    def update_gui(self):
        # 1. EMPTY GUI UPDATE QUEUE
        with self.history_lock:
            while not self.gui_update_queue.empty():
                try: t_chunk, data_chunk_dict, math_vals = self.gui_update_queue.get_nowait()
                except queue.Empty: break
                
                self.history_time.extend(t_chunk)
                for sig in self.available_signals:
                    if sig in data_chunk_dict:
                        self.history_data[sig].extend(data_chunk_dict[sig])
                self.latest_math_values.update(math_vals)

        # 2. UPDATE FPS
        now = time.perf_counter()
        dt = now - self.last_update_time
        self.last_update_time = now
        if dt > 0: self.fps_queue.append(1.0 / dt)
        if len(self.fps_queue) > 0: self.fps_label.setText(f"GUI FPS: {sum(self.fps_queue)/len(self.fps_queue):.1f}")

        # 3. UPDATE MATH INDICATORS
        current_configs = self.active_channel_configs
        units = {cfg["Name"]: cfg["Unit"] for cfg in current_configs}
        custom_names = {cfg["Name"]: cfg["CustomName"] for cfg in current_configs}
        units["DMM"] = "V" 
        custom_names["DMM"] = "DMM"
        
        math_sigs = set()
        for ind in self.indicator_widgets:
            raw_sig = ind.signal_cb.currentData()
            math_sigs.add(raw_sig)
            ctype = ind.type_cb.currentText()
            val = self.latest_math_values.get(raw_sig, {}).get(ctype, 0.0)
            ind.update_display(val, units.get(raw_sig, "V"), raw_sig)
        self.active_math_signals = math_sigs

        # 4. PREPARE PLOT DATA
        try: window_s = float(self.plot_window_input.text())
        except ValueError: window_s = 10.0

        required_signals = set()
        for widget in self.subplot_widgets:
            for sig in widget.get_selected_signals(): required_signals.add(sig)

        with self.history_lock:
            if len(self.history_time) == 0: return
            t_full = np.array(self.history_time)
            y_full = {sig: np.array(self.history_data[sig]) for sig in required_signals}

        if window_s > 0 and len(t_full) > 0:
            cutoff_time = t_full[-1] - window_s
            start_idx = np.searchsorted(t_full, cutoff_time)
            t_arr = t_full[start_idx:]
            y_arrays = {sig: y_full[sig][start_idx:] for sig in required_signals}
        else:
            t_arr = t_full
            y_arrays = y_full

        # FAST VISUAL DECIMATION (Prevents Python-to-C++ memory traffic jams)
        # Limits data to 5000 points (slightly more than a 4K monitor can display)
        MAX_VISUAL_POINTS = 5000 
        step = max(1, len(t_arr) // MAX_VISUAL_POINTS)
        
        t_arr = t_arr[::step]
        for sig in y_arrays:
            y_arrays[sig] = y_arrays[sig][::step]

        # NO VISUAL DECIMATION NEEDED WITH PYQTGRAPH, it handles full arrays flawlessly

        # 5. REBUILD PYQTGRAPH IF NEEDED
        if self.needs_plot_rebuild:
            # 1. Properly destroy old ghost plots so they don't freeze in the background
            if hasattr(self, 'plots'):
                for p in self.plots:
                    p.close()
                    p.deleteLater()
            
            self.graph_layout.clear()
            self.graph_layout.ci.layout.setSpacing(15) # Clean 15px gap between plots
            
            self.plots = []
            self.curves = {}
            n = len(self.subplot_widgets)
            
            if n > 0:
                for i in range(n):
                    p = self.graph_layout.addPlot(row=i, col=0)
                    p.showGrid(x=True, y=True, alpha=0.3)
                    
                    # 2. Link X-axes so zooming/panning one zooms all of them seamlessly!
                    if i > 0:
                        p.setXLink(self.plots[0])
                    
                    # 3. Hide redundant X-axis text on the upper plots to prevent overlap
                    if i < n - 1:
                        p.showAxis('bottom', False)
                    else:
                        p.setLabel('bottom', "Time", units="s")
                    
                    self.plots.append(p)
                    self.curves[i] = {}
                    selected_raw_signals = self.subplot_widgets[i].get_selected_signals()
                    plot_units = set()
                    
                    for idx_color, raw_sig in enumerate(selected_raw_signals):
                        if raw_sig not in self.available_signals: continue
                        unit = units.get(raw_sig, "V")
                        c_name = custom_names.get(raw_sig, raw_sig)
                        plot_units.add(unit)
                        
                        color_tuple = PLOT_COLORS[idx_color % len(PLOT_COLORS)]
                        pen = pg.mkPen(color=color_tuple, width=1.5)
                        
                        curve = p.plot(pen=pen, name=f"{c_name} [{unit}]")
                        self.curves[i][raw_sig] = curve
                    
                    if plot_units: p.setLabel('left', f"Value [{', '.join(sorted(list(plot_units)))}]")
                    else: p.setLabel('left', "Value")
                    
                    # Pin legend inside the plot cleanly
                    if selected_raw_signals: 
                        p.addLegend(offset=(10, 10))
            
            self.needs_plot_rebuild = False

        # 6. UPDATE PYQTGRAPH DATA
        for i, p in enumerate(self.plots):
            for raw_sig in self.subplot_widgets[i].get_selected_signals():
                if raw_sig in self.curves[i] and raw_sig in y_arrays:
                    y_arr = y_arrays[raw_sig]
                    # Update line instantly
                    self.curves[i][raw_sig].setData(t_arr, y_arr)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.folder_display.setText(f"Output Folder: {self.output_folder[-40:]}")

    def generate_filename(self, base_name=""):
        timestamp = datetime.now().strftime("%H-%M-%S")
        name = f"{base_name.strip()}_{timestamp}.tdms" if base_name.strip() else f"{timestamp}.tdms"
        date_str = datetime.now().strftime("%d_%m_%Y")
        folder_path = Path(self.output_folder) / date_str
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path / name

    def generate_profile(self, write_rate): return [] 

    def start_write(self):
        if self.start_timestamp is None: self.start_timestamp = time.time()
        self.write_stop_flag.clear()
        
        if self.simulate_checkbox.isChecked():
            self.write_active_label.setStyleSheet("color: green; font-weight: bold;")
            self.shutdown_label.setText("Status: OK (Simulated)")
            self.shutdown_label.setStyleSheet("color: green; font-weight: bold;")
            return

        write_rate = float(self.write_rate_input.text())
        current_profile = self.generate_profile(write_rate)
        voltages = [val / 200.0 for val in current_profile] 
        if voltages and voltages[-1] != 0.0: voltages.append(0.0)

        with self.write_task_lock:
            if self.write_task is not None:
                try: self.write_task.stop(); self.write_task.close()
                except: pass
            
            write_channel = self.get_write_channel()
            self.write_task = nidaqmx.Task()
            self.write_task.ao_channels.add_ao_voltage_chan(write_channel)
            self.write_task.timing.cfg_samp_clk_timing(write_rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(voltages))
            try: self.write_task.write(voltages, auto_start=True)
            except nidaqmx.errors.DaqError:
                try: self.write_task.close()
                except: pass
                time.sleep(0.1)
                self.write_task = nidaqmx.Task()
                self.write_task.ao_channels.add_ao_voltage_chan(write_channel)
                self.write_task.timing.cfg_samp_clk_timing(write_rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(voltages))
                self.write_task.write(voltages, auto_start=True)

        self.write_active_label.setStyleSheet("color: green; font-weight: bold;")
        self.shutdown_label.setText("Status: OK")
        self.shutdown_label.setStyleSheet("color: green; font-weight: bold;")

    def stop_write(self):
        self.write_stop_flag.set()
        self.write_active_label.setStyleSheet("color: grey; font-weight: bold;")
        if self.simulate_checkbox.isChecked(): return
        with self.write_task_lock:
            if self.write_task is not None:
                try: self.write_task.stop(); self.write_task.close()
                except: pass
                self.write_task = None
        try: self.zero_ao_output()
        except: pass

    def zero_ao_output(self):
        if self.simulate_checkbox.isChecked(): return
        try:
            with nidaqmx.Task() as zero_task:
                zero_task.ao_channels.add_ao_voltage_chan(self.get_write_channel())
                zero_task.write(0.0); time.sleep(0.01); zero_task.write(0.0)
        except: pass

    def reset_all(self):
        self.write_stop_flag.set()
        self.mp_stop_flag.set()
        self.gui_timer.stop()
        
        self.read_rate_input.setEnabled(True)
        self.average_samples_input.setEnabled(True)

        if self.daq_process and self.daq_process.is_alive(): self.daq_process.join(timeout=3)
        if self.math_process and self.math_process.is_alive(): self.math_process.join(timeout=3)
        if self.tdms_writer_thread and self.tdms_writer_thread.is_alive(): self.tdms_writer_thread.join(timeout=3)

        with self.write_task_lock:
            if self.write_task is not None:
                try: self.write_task.stop(); self.write_task.close()
                except: pass
                self.write_task = None

        self.zero_ao_output()
        with self.history_lock:
            self.history_time.clear()
            for sig in self.available_signals: self.history_data[sig].clear()

        self.write_active_label.setStyleSheet("color: grey; font-weight: bold;")
        self.shutdown_label.setText("Status: OK")
        self.shutdown_label.setStyleSheet("color: green; font-weight: bold;")
        self.start_timestamp = None

        for ind in self.indicator_widgets: ind.value_label.setText("0.00")
        self.needs_plot_rebuild = True
        
        self.last_update_time = time.perf_counter()
        self.fps_queue.clear()
        self.fps_label.setText("GUI FPS: 0.0")
        self.update_gui()

    def open_export_dialog(self):
        self.export_ascii_btn.setEnabled(False)
        dialog = ExportDialog(self)
        dialog.exec_()
        self.export_ascii_btn.setEnabled(True)

    def closeEvent(self, event):
        print("[INFO] Exiting application. Saving config, stopping tasks and zeroing AO0...")
        self.save_config()
        self.write_stop_flag.set()
        self.mp_stop_flag.set()
        self.gui_timer.stop()
        
        if hasattr(self, 'ao_window') and self.ao_window is not None: self.ao_window.close() 
        
        if self.daq_process and self.daq_process.is_alive(): self.daq_process.terminate()
        if self.math_process and self.math_process.is_alive(): self.math_process.terminate()
        if self.tdms_writer_thread and self.tdms_writer_thread.is_alive(): self.tdms_writer_thread.join(timeout=2)
        
        with self.write_task_lock:
            if self.write_task is not None:
                try: self.write_task.stop(); self.write_task.close()
                except: pass
        self.zero_ao_output()
        event.accept()

    def exit_application(self): self.close() 

if __name__ == "__main__":
    mp.freeze_support() # REQUIRED FOR WINDOWS MULTIPROCESSING
    app = QApplication(sys.argv)
    gui = DAQControlApp()
    gui.show()
    sys.exit(app.exec_())