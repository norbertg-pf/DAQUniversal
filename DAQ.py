import sys
import os
import json
import logging
import threading
import multiprocessing as mp
import queue
import socket
import numpy as np
import nidaqmx
import nidaqmx.system
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, LineGrouping, ThermocoupleType, CJCSource

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QGridLayout,
                             QCheckBox, QFileDialog, QScrollArea,
                             QTabWidget, QComboBox, QMessageBox, QGroupBox, QFrame)
from PyQt5.QtCore import QTimer, Qt

# --- REPLACED MATPLOTLIB WITH PYQTGRAPH ---
import pyqtgraph as pg
pg.setConfigOption('background', 'w')  # Set white background to match Matplotlib
pg.setConfigOption('foreground', 'k')  # Set black axes/text
pg.setConfigOption('antialias', False)  # Keep real-time rendering fast

from nptdms import TdmsWriter, ChannelObject, TdmsFile
from datetime import datetime
import collections
import time
from pathlib import Path
import traceback

from hardware_profiles import DEFAULT_PROFILE_NAME, detect_profile_name, get_profile

from app_constants import ALL_AO_CHANNELS, ALL_CHANNELS, PLOT_COLORS

# Global Constants
DEFAULT_HARDWARE_PROFILE = get_profile(DEFAULT_PROFILE_NAME)

from processing_workers import get_terminal_name_with_dev_prefix, daq_read_worker, math_processing_worker

from cloud_upload import UploadThread
from pulse_controls import VoltageToggleWindow
from ui_components import (
    ChannelSelectionDialog,
    ExportDialog,
    GDriveHelpDialog,
    MathHelpDialog,
    NumericalIndicatorWidget,
    SubplotConfigWidget,
)



class DAQControlApp(QWidget):
    MATH_DEVICE_ID = "__MATH_DEVICE__"
    ALL_DEVICES_ID = "__ALL_DEVICES__"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAQ Control GUI")
        self.logger = logging.getLogger(__name__)
        self._error_log_times = {}

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

        # Simulation mode is intentionally always enabled.
        self.simulate_mode = True
        self.history_lock = threading.Lock() 

        self.start_timestamp = None
        self.sample_nr = 0
        self.output_folder = r"\data"
        self.current_tdms_filepath = ""
        self.export_settings = {}
        
        self.current_hardware_profile = DEFAULT_HARDWARE_PROFILE
        self.device_product_types = {}
        self.detected_devices = []
        self.available_signals = self.current_hardware_profile.default_enabled_signals.copy()
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
            if dev_name and dev_name not in (self.MATH_DEVICE_ID, self.ALL_DEVICES_ID):
                nidaqmx.system.Device(dev_name).reset_device()
                do_init = nidaqmx.Task()
                do_init.do_channels.add_do_chan(self.get_do_channel(), line_grouping=LineGrouping.CHAN_PER_LINE)
                do_init.write([False, False])
                do_init.close()
        except Exception as e:
            print(f"[WARN] DAQ Hardware not found at startup: {e}.")

    def _should_log_error(self, key, interval_sec=5.0):
        now = time.monotonic()
        last = self._error_log_times.get(key, 0.0)
        if now - last >= interval_sec:
            self._error_log_times[key] = now
            return True
        return False

    def _log_exception(self, context, exc, key=None, interval_sec=5.0):
        error_key = key or context
        if not self._should_log_error(error_key, interval_sec=interval_sec):
            return
        msg = f"[ERROR] {context}: {exc}"
        try:
            self.logger.exception(msg)
        except Exception:
            print(msg)
            print(traceback.format_exc())

    def _set_shutdown_status(self, text, color="green"):
        def _update():
            self.shutdown_label.setText(text)
            self.shutdown_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        QTimer.singleShot(0, _update)

    def _show_fatal_error(self, title, message):
        def _show():
            QMessageBox.critical(self, title, message)
        QTimer.singleShot(0, _show)

    def get_default_channel_config(self, raw_name):
        base = self._signal_base_name(raw_name)
        term = "RSE" if base.startswith("AI") and self._ai_index(base) >= 16 else "DIFF"
        return {
            "custom_name": raw_name, "term": term, "range": "-10 to 10",
            "sensor": "None", "scale": "1.0", "unit": "V", "offset": "0.0",
            "lpf_on": False, "lpf_cutoff": "10.0", "lpf_order": "4",
            "expression": "AI0 - AI1"
        }

    def _is_math_device_selected(self):
        return self.device_cb.currentData() == self.MATH_DEVICE_ID

    def _is_all_devices_selected(self):
        return self.device_cb.currentData() == self.ALL_DEVICES_ID

    def _signal_base_name(self, signal_name):
        return signal_name.split("@", 1)[0] if "@" in signal_name else signal_name

    def _signal_device_name(self, signal_name):
        return signal_name.split("@", 1)[1] if "@" in signal_name else None

    def _is_ai_signal(self, signal_name):
        return self._signal_base_name(signal_name).startswith("AI")

    def _is_ao_signal(self, signal_name):
        return self._signal_base_name(signal_name).startswith("AO")

    def _ai_index(self, ai_name):
        digits = "".join(ch for ch in ai_name[2:] if ch.isdigit())
        return int(digits) if digits else 0

    def _mk_dev_signal(self, device_name, raw_name):
        return f"{raw_name}@{device_name}"

    def _channels_for_device(self, device_name):
        ptype = self.device_product_types.get(device_name, "")
        profile = get_profile(detect_profile_name(ptype, device_name))
        return [self._mk_dev_signal(device_name, sig) for sig in (profile.ai_channels + profile.ao_channels)]

    def _ensure_signal_state(self, signal_name):
        if signal_name not in self.master_channel_configs:
            self.master_channel_configs[signal_name] = self.get_default_channel_config(signal_name)
        if signal_name not in self.history_data:
            self.history_data[signal_name] = collections.deque(maxlen=self.history_maxlen)

    def _get_active_hw_device(self):
        dev_name = self.device_cb.currentData()
        if dev_name and dev_name not in (self.MATH_DEVICE_ID, self.ALL_DEVICES_ID):
            return dev_name
        for i in range(self.device_cb.count()):
            candidate = self.device_cb.itemData(i)
            if candidate not in (self.MATH_DEVICE_ID, self.ALL_DEVICES_ID):
                return candidate
        return "Dev1"

    def get_write_channel(self):
        return f"{self._get_active_hw_device()}/ao0"

    def get_do_channel(self):
        return f"{self._get_active_hw_device()}/port0/line0:1"

    def update_ao_state_from_pulse(self, channels, volts):
        for ch in channels:
            raw_sig = ch.split('/')[-1].upper() 
            if raw_sig in self.ao_state_dict:
                self.ao_state_dict[raw_sig] = float(volts)

    def launch_analog_out(self):
        device_name = self._get_active_hw_device()
        active_aos = [self._signal_base_name(s) for s in self.available_signals if self._is_ao_signal(s)]
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
        self.fps_label = QLabel("GUI FPS: 0.0")
        self.fps_label.setStyleSheet("color: gray; font-weight: bold;")
        self.simulation_note_label = QLabel("SIMULATION MODE")
        self.simulation_note_label.setStyleSheet("color: red; font-weight: bold;")
        
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
        self.open_ao_btn = QPushButton("Open AnalogOut Control")
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
        status_row.addWidget(self.upload_status_label)
        status_row.addSpacing(12)
        status_row.addWidget(self.simulation_note_label)
        status_row.addStretch()
        controls_layout.addLayout(status_row, 7, 0, 1, 2)
        
        plot_control_layout = QVBoxLayout()
        plot_header = QHBoxLayout()
        plot_header.addWidget(QLabel("<b>Dynamic Plot Configuration</b>"))
        plot_header.addStretch()
        plot_header.addWidget(self.fps_label)
        plot_control_layout.addLayout(plot_header)
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
        self.graph_layout.setViewportUpdateMode(self.graph_layout.MinimalViewportUpdate)
        self.graph_layout.setStyleSheet("background-color: #f7f9fc; border: 1px solid #d9dee8; border-radius: 6px;")
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
        self.device_product_types = {}
        self.detected_devices = []
        devs = []
        try:
            sys_local = nidaqmx.system.System.local()
            for d in sys_local.devices:
                devs.append((d.name, d.product_type))
            if not devs:
                devs = [("Dev1", "Simulated Device")]
        except Exception:
            devs = [("Dev1", "Simulated Device")]

        idx_to_set = 0
        for i, (name, ptype) in enumerate(devs):
            self.device_product_types[name] = ptype
            self.detected_devices.append(name)
            self.device_cb.addItem(f"{name} ({ptype})", userData=name)
            if name == current_data:
                idx_to_set = i

        self.device_cb.addItem("All Devices", userData=self.ALL_DEVICES_ID)
        self.device_cb.addItem("Math (Virtual)", userData=self.MATH_DEVICE_ID)

        if self.device_cb.count() > 0:
            self.device_cb.setCurrentIndex(idx_to_set)
            self.on_device_changed(self.device_cb.currentIndex())

    def get_selected_device_profile(self):
        if self._is_math_device_selected() or self._is_all_devices_selected():
            return DEFAULT_HARDWARE_PROFILE

        dev_name = self.device_cb.currentData() or ""
        product_type = self.device_product_types.get(dev_name, "")
        profile_name = detect_profile_name(product_type, dev_name)
        return get_profile(profile_name)

    def apply_selected_device_profile(self):
        if self._is_math_device_selected():
            kept_math = [s for s in self.available_signals if s.startswith("MATH")]
            self.available_signals = kept_math if kept_math else [f"MATH{i}" for i in range(4)]
            for sig in self.available_signals: self._ensure_signal_state(sig)
            return

        if self._is_all_devices_selected():
            valid_hw = set()
            for dev in self.detected_devices:
                valid_hw.update(self._channels_for_device(dev))
            valid_hw.add("DMM")
            kept_math = [s for s in self.available_signals if s.startswith("MATH")]
            kept_hw = [s for s in self.available_signals if s in valid_hw]
            if not kept_hw and self.detected_devices:
                kept_hw = self._channels_for_device(self.detected_devices[0])
            self.available_signals = kept_hw + kept_math
            for sig in self.available_signals: self._ensure_signal_state(sig)
            return

        self.current_hardware_profile = self.get_selected_device_profile()
        dev = self.device_cb.currentData()
        valid_hw = set(self._channels_for_device(dev) + ["DMM"])

        # Keep virtual channels, clamp hardware channels to current device profile.
        kept_math = [s for s in self.available_signals if s.startswith("MATH")]
        kept_hw = [s for s in self.available_signals if s in valid_hw]
        if not kept_hw:
            kept_hw = [self._mk_dev_signal(dev, sig) for sig in self.current_hardware_profile.default_enabled_signals]

        self.available_signals = kept_hw + kept_math
        for sig in self.available_signals: self._ensure_signal_state(sig)

    def on_device_changed(self, _):
        if not hasattr(self, "config_grid"):
            return
        self.apply_selected_device_profile()
        new_device = self.device_cb.currentData()
        if not new_device:
            return
        for ch in getattr(self, "channel_ui_configs", []):
            raw_name = ch['name']
            if not raw_name.startswith("MATH"):
                ch['ch_label'].setText(f"{new_device}/{raw_name.lower()} ({raw_name})")
        self.rebuild_config_tab()
    def open_channel_selector(self):
        self.cache_current_ui_configs()
        if self._is_math_device_selected():
            allowed_signals = [f"MATH{i}" for i in range(4)]
            active_signals = [s for s in self.available_signals if s.startswith("MATH")]
        elif self._is_all_devices_selected():
            allowed_signals = []
            for dev in self.detected_devices:
                allowed_signals.extend(self._channels_for_device(dev))
            allowed_signals.append("DMM")
            active_signals = [s for s in self.available_signals if s in allowed_signals]
        else:
            dev = self.device_cb.currentData()
            allowed_signals = self._channels_for_device(dev) + ["DMM"]
            active_signals = [s for s in self.available_signals if s in allowed_signals]

        dialog = ChannelSelectionDialog(active_signals, self, allowed_signals=allowed_signals)
        if dialog.exec_():
            selected = dialog.get_selected()
            if self._is_math_device_selected():
                self.available_signals = selected
            else:
                math_signals = [s for s in self.available_signals if s.startswith("MATH")]
                self.available_signals = selected + math_signals
            for sig in self.available_signals: self._ensure_signal_state(sig)
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
            if self._is_ai_signal(ch_ui['name']):
                idx = self._ai_index(self._signal_base_name(ch_ui['name']))
                ch_ui['range_cb'].setCurrentText(b_range)
                ch_ui['sensor_cb'].setCurrentText(b_sensor)
                if idx >= 16 and b_term == "DIFF": ch_ui['term_cb'].setCurrentText("RSE")
                else: ch_ui['term_cb'].setCurrentText(b_term)

    def calibrate_single_offset(self, raw_name, scale_input, offset_input, term_cb, range_cb, sensor_cb):
        if self.simulate_mode:
            offset_input.setText("0.000")
            return
        try: scale_val = float(scale_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid Scale Factor.")
            return
        try:
            with nidaqmx.Task() as task:
                dev_prefix = self._get_active_hw_device()
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
        self.device_cb.currentIndexChanged.connect(self.on_device_changed)
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

        # Populate devices only after config widgets/layouts exist
        self.refresh_devices()
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
        dev_prefix = self._get_active_hw_device()
        import functools
        
        active_ai = [s for s in self.available_signals if self._is_ai_signal(s)]
        if active_ai:
            ai_heading = "Math Device Inputs" if self._is_math_device_selected() else "Analog Inputs (AI)"
            ai_label = QLabel(f"<b>{ai_heading}</b>")
            ai_label.setStyleSheet("color: #0055a4; font-size: 14px; margin-top: 10px; margin-bottom: 5px;")
            self.config_grid.addWidget(ai_label, row_idx, 0, 1, 12)
            row_idx += 1

        for raw_name in self.available_signals:
            if not self._is_ai_signal(raw_name): continue

            cfg = self.master_channel_configs[raw_name]
            base_name = self._signal_base_name(raw_name)
            dev_name = self._signal_device_name(raw_name) or dev_prefix
            ch_label = QLabel(f"{dev_name}/{base_name.lower()} ({raw_name})")
            custom_name_input = QLineEdit(cfg.get("custom_name", raw_name))
            term_cb = QComboBox()
            if self._ai_index(base_name) >= 16:
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

        active_ao = [s for s in self.available_signals if self._is_ao_signal(s)]
        if active_ao:
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            self.config_grid.addWidget(line, row_idx, 0, 1, 12)
            row_idx += 1
            ao_heading = "Math Device Outputs" if self._is_math_device_selected() else "Analog Outputs (AO)"
            ao_label = QLabel(f"<b>{ao_heading}</b>")
            ao_label.setStyleSheet("color: #0055a4; font-size: 14px; margin-top: 5px; margin-bottom: 5px;")
            self.config_grid.addWidget(ao_label, row_idx, 0, 1, 12)
            row_idx += 1
            
            for raw_name in active_ao:
                cfg = self.master_channel_configs[raw_name]
                base_name = self._signal_base_name(raw_name)
                dev_name = self._signal_device_name(raw_name) or dev_prefix
                ch_label = QLabel(f"{dev_name}/{base_name.lower()} ({raw_name})")
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
            m_title = "Math Device Channels" if self._is_math_device_selected() else "Virtual Math Channels"
            m_label = QLabel(f"<b>{m_title}</b>")
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
                "window_width": self.width(),
                "window_height": self.height(),
                "threshold": self.threshold_input.text(),
                "dmm_ip": self.Keithley_DMM_IP.text(),
                "gdrive_link": self.gdrive_link_input.text(),
                "gdrive_auth": self.gdrive_auth_cb.currentText(),
                "simulate": self.simulate_mode,
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
            if "window_width" in main_cfg and "window_height" in main_cfg:
                try: self.resize(int(main_cfg["window_width"]), int(main_cfg["window_height"]))
                except (TypeError, ValueError): pass
            if "threshold" in main_cfg: self.threshold_input.setText(main_cfg["threshold"])
            if "dmm_ip" in main_cfg: self.Keithley_DMM_IP.setText(main_cfg["dmm_ip"])
            if "gdrive_link" in main_cfg: self.gdrive_link_input.setText(main_cfg["gdrive_link"])
            
            if "gdrive_auth" in main_cfg: 
                idx = self.gdrive_auth_cb.findText(main_cfg["gdrive_auth"])
                if idx >= 0: self.gdrive_auth_cb.setCurrentIndex(idx)

            if "output_folder" in main_cfg: 
                self.output_folder = main_cfg["output_folder"]
                self.folder_display.setText(f"Output Folder: ..{self.output_folder[-30:]}")

            if "available_signals" in main_cfg: self.available_signals = main_cfg["available_signals"]
            for sig in self.available_signals: self._ensure_signal_state(sig)
            self.apply_selected_device_profile()
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

        dev_prefix = self._get_active_hw_device()
        daq_configs = []
        for ch in self.channel_ui_configs:
            if ch['name'].startswith("MATH"):
                daq_configs.append({
                    'Name': ch['name'],
                    'CustomName': ch['custom_name_input'].text().strip() or ch['name'],
                    'Expression': ch['expr_input'].text().strip(),
                    'Unit': ch['unit_input'].text().strip(),
                    'Kind': 'MATH',
                    'Scale': 1.0, 'Offset': 0.0 
                })
                continue

            sensor_type = ch['sensor_cb'].currentText() if 'sensor_cb' in ch else "None"
            base_name = self._signal_base_name(ch['name'])
            dev_name = self._signal_device_name(ch['name']) or dev_prefix
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
                'Name': ch['name'], 'Terminal': f"{dev_name}/{base_name.lower()}",
                'CustomName': ch['custom_name_input'].text().strip() or ch['name'],
                'Config': config_map.get(t_cb_text, TerminalConfiguration.RSE),
                'Range': parse_range(r_cb_text),
                'SensorType': sensor_type, 'Scale': scale_val,
                'Unit': ch['unit_input'].text().strip(), 'Offset': offset_val,
                'Kind': 'AI' if self._is_ai_signal(ch['name']) else 'AO',
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
        active_ai_configs = [c for c in configs if c.get('Kind') == 'AI']
        hw_signals = [sig for sig in self.available_signals if not sig.startswith("MATH")]
        math_signals = [sig for sig in self.available_signals if sig.startswith("MATH")]
        
        try: read_rate = float(self.read_rate_input.text())
        except ValueError: read_rate = 10000.0
        try: average_samples = max(1, int(self.average_samples_input.text()))
        except ValueError: average_samples = 100
        samples_per_read = max(1, int(read_rate // 10)) 
        
        simulate = self.simulate_mode
        has_dmm = "DMM" in self.available_signals
        n_ai = len(active_ai_configs)
        active_ao_signals = [c['Name'] for c in configs if c.get('Kind') == 'AO']
        n_ao = len(active_ao_signals)
        
        self.daq_process = mp.Process(target=daq_read_worker, args=(
            self.mp_stop_flag, simulate, read_rate, samples_per_read, active_ai_configs, 
            n_ai, n_ao, active_ao_signals, has_dmm, self.available_signals, self.ao_state_dict, self.dmm_buffer_list, 
            self.tdms_queue, self.process_queue))
            
        self.math_process = mp.Process(target=math_processing_worker, args=(
            self.mp_stop_flag, read_rate, average_samples, self.available_signals, 
            hw_signals, math_signals, cfg_dict, self.process_queue, self.gui_update_queue))

        time.sleep(0.1)
        self.daq_process.start()
        self.math_process.start()
        
        self.last_update_time = time.perf_counter()
        self.gui_timer.start(int(1000 / 15))

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
        active_ai = [s for s in self.available_signals if self._is_ai_signal(s)]
        active_ao = [s for s in self.available_signals if self._is_ao_signal(s)]
        has_dmm = "DMM" in self.available_signals

        try:
            with TdmsWriter(str(filename)) as writer:
                while not self.mp_stop_flag.is_set() or not self.tdms_queue.empty():
                    try:
                        chunk = self.tdms_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    except (EOFError, OSError, ValueError) as exc:
                        self._log_exception("TDMS queue read failure", exc, key="tdms_queue_read", interval_sec=10.0)
                        continue

                    try:
                        time_arr, ai_data, ao_data, dmm_data = chunk
                        channels = [ChannelObject("RawData", "Time", time_arr)]
                        for i, sig in enumerate(active_ai):
                            channels.append(ChannelObject("RawData", sig, ai_data[i]))
                        for i, sig in enumerate(active_ao):
                            channels.append(ChannelObject("RawData", sig, ao_data[i]))
                        if has_dmm:
                            channels.append(ChannelObject("RawData", "DMM", dmm_data.flatten()))
                        writer.write_segment(channels)
                    except (IndexError, TypeError, ValueError, AttributeError) as exc:
                        self._log_exception("Failed to write TDMS segment", exc, key="tdms_segment_write", interval_sec=10.0)
        except (OSError, IOError, ValueError) as exc:
            self._log_exception("TDMS writer failed", exc, key="tdms_writer_fatal", interval_sec=10.0)
            self._set_shutdown_status("Status: TDMS write error", color="red")

    def DMM_read(self):
        if "DMM" not in self.available_signals:
            return
        if self.simulate_mode:
            while not self.mp_stop_flag.is_set():
                self.dmm_buffer_list.append(float(np.random.uniform(-0.1, 0.1)))
                time.sleep(0.1)
            return

        inst = None
        try:
            inst = DMM6510readout.write_script_to_Keithley(self.Keithley_DMM_IP.text(), "0.05")
            while not self.mp_stop_flag.is_set():
                try:
                    self.dmm_buffer_list.append(float(DMM6510readout.read_data(inst)))
                except (TypeError, ValueError) as exc:
                    self._log_exception("Invalid DMM data received", exc, key="dmm_read_data", interval_sec=10.0)
                    time.sleep(0.05)
                except (OSError, RuntimeError) as exc:
                    self._log_exception("DMM read transport error", exc, key="dmm_transport", interval_sec=10.0)
                    time.sleep(0.1)
        except (OSError, RuntimeError, ValueError) as exc:
            self._log_exception("Failed to initialize DMM readout", exc, key="dmm_init", interval_sec=10.0)
            self._set_shutdown_status("Status: DMM read failure", color="red")
        finally:
            if inst is not None:
                try:
                    DMM6510readout.stop_instrument(inst)
                except (OSError, RuntimeError, AttributeError) as exc:
                    self._log_exception("Failed to stop DMM instrument", exc, key="dmm_stop", interval_sec=30.0)

    def update_gui(self):
        # 1. EMPTY GUI UPDATE QUEUE
        with self.history_lock:
            while not self.gui_update_queue.empty():
                try: t_chunk, data_chunk_dict, math_vals = self.gui_update_queue.get_nowait()
                except queue.Empty: break
                
                self.history_time.extend(t_chunk)
                for sig in self.available_signals:
                    if sig in data_chunk_dict:
                        self._ensure_signal_state(sig)
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
            self.graph_layout.clear()
            self.graph_layout.ci.layout.setSpacing(10)
            self.graph_layout.ci.layout.setContentsMargins(12, 12, 12, 12)
            
            self.plots = []
            self.curves = {}
            n = len(self.subplot_widgets)
            
            if n > 0:
                for i in range(n):
                    p = self.graph_layout.addPlot(row=i, col=0)
                    p.showGrid(x=True, y=True, alpha=0.3)
                    p.setClipToView(True)
                    p.getViewBox().setDefaultPadding(0.02)
                    p.getAxis('left').setWidth(80)
                    p.setMinimumHeight(180)
                    
                    # 2. Link X-axes so zooming/panning one zooms all of them seamlessly!
                    if i > 0:
                        p.setXLink(self.plots[0])
                    
                    # 3. Hide redundant X-axis text on the upper plots to prevent overlap
                    if i < n - 1:
                        p.showAxis('bottom', False)
                        p.getAxis('bottom').setHeight(0)
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
        
        if self.simulate_mode:
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
                try:
                    self.write_task.stop()
                    self.write_task.close()
                except (nidaqmx.errors.DaqError, RuntimeError, AttributeError) as exc:
                    self._log_exception("Failed to stop existing AO write task", exc, key="ao_stop_existing", interval_sec=10.0)

            write_channel = self.get_write_channel()
            self.write_task = nidaqmx.Task()
            self.write_task.ao_channels.add_ao_voltage_chan(write_channel)
            self.write_task.timing.cfg_samp_clk_timing(write_rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(voltages))
            try: self.write_task.write(voltages, auto_start=True)
            except nidaqmx.errors.DaqError as exc:
                self._log_exception("Initial AO write failed, retrying", exc, key="ao_write_retry", interval_sec=10.0)
                try:
                    self.write_task.close()
                except (nidaqmx.errors.DaqError, RuntimeError, AttributeError) as close_exc:
                    self._log_exception("Failed to close AO task before retry", close_exc, key="ao_close_before_retry", interval_sec=10.0)
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
        if self.simulate_mode: return
        with self.write_task_lock:
            if self.write_task is not None:
                try:
                    self.write_task.stop()
                    self.write_task.close()
                except (nidaqmx.errors.DaqError, RuntimeError, AttributeError) as exc:
                    self._log_exception("Failed to stop AO write task", exc, key="ao_stop", interval_sec=10.0)
                self.write_task = None
        try:
            self.zero_ao_output()
        except (nidaqmx.errors.DaqError, RuntimeError, AttributeError) as exc:
            self._log_exception("Failed to zero AO output", exc, key="ao_zero", interval_sec=10.0)
            self._set_shutdown_status("Status: AO shutdown warning", color="orange")

    def zero_ao_output(self):
        if self.simulate_mode:
            return
        try:
            with nidaqmx.Task() as zero_task:
                zero_task.ao_channels.add_ao_voltage_chan(self.get_write_channel())
                zero_task.write(0.0)
                time.sleep(0.01)
                zero_task.write(0.0)
        except (nidaqmx.errors.DaqError, RuntimeError, AttributeError) as exc:
            self._log_exception("Zero AO output failed", exc, key="ao_zero_direct", interval_sec=10.0)

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
                try:
                    self.write_task.stop()
                    self.write_task.close()
                except (nidaqmx.errors.DaqError, RuntimeError, AttributeError) as exc:
                    self._log_exception("Failed to stop AO task during reset", exc, key="ao_reset_stop", interval_sec=10.0)
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
                try:
                    self.write_task.stop()
                    self.write_task.close()
                except (nidaqmx.errors.DaqError, RuntimeError, AttributeError) as exc:
                    self._log_exception("Failed to stop AO task during close", exc, key="ao_close_stop", interval_sec=10.0)
                    self._set_shutdown_status("Status: AO close warning", color="orange")
                    self._show_fatal_error("Shutdown Warning", "Failed to stop AO task cleanly. Hardware may require manual reset.")
        self.zero_ao_output()
        event.accept()

    def exit_application(self): self.close() 

if __name__ == "__main__":
    mp.freeze_support() # REQUIRED FOR WINDOWS MULTIPROCESSING
    app = QApplication(sys.argv)
    gui = DAQControlApp()
    gui.show()
    sys.exit(app.exec_())
