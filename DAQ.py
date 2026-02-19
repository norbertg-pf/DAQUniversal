import sys
import os
import json
import threading
import queue
import subprocess
import socket
import nidaqmx.system
import numpy as np
import nidaqmx
from nidaqmx import stream_readers
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, LineGrouping, ProductCategory, ThermocoupleType, CJCSource
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QGridLayout,
                             QCheckBox, QListWidget, QListWidgetItem, QFileDialog, QScrollArea,
                             QTabWidget, QComboBox, QDialog, QMessageBox, QGroupBox, QDoubleSpinBox, QFormLayout, QFrame)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread

# Using Matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector

from nptdms import TdmsWriter, ChannelObject, TdmsFile
from datetime import datetime
import collections
import time
from pathlib import Path
import traceback

ALL_AI_CHANNELS = [f"AI{i}" for i in range(32)]
ALL_AO_CHANNELS = [f"AO{i}" for i in range(4)]
ALL_CHANNELS = ALL_AI_CHANNELS + ALL_AO_CHANNELS + ["DMM"]

def get_terminal_name_with_dev_prefix(task: nidaqmx.Task, terminal_name: str) -> str:
    for device in task.devices:
        if device.product_category not in [ProductCategory.C_SERIES_MODULE, ProductCategory.SCXI_MODULE]:
            return f"/{device.name}/{terminal_name}"
    raise RuntimeError("Suitable device not found in task.")

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

    def stop(self):
        self._stop = True

    def _write_all(self, task, value: float):
        if len(self.channels) > 1:
            task.write([value] * len(self.channels))
        else:
            task.write(value)
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
        if state == "HIGH":
            self.status_label.setStyleSheet("background-color: #28a745; color: white; font-size: 13px; padding: 6px; border-radius: 4px;")
        elif state == "LOW":
            self.status_label.setStyleSheet("background-color: #6c757d; color: white; font-size: 13px; padding: 6px; border-radius: 4px;")
        elif state == "ERROR":
            self.status_label.setStyleSheet("background-color: #dc3545; color: white; font-size: 13px; padding: 6px; border-radius: 4px;")

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
            lbl.setStyleSheet("color: gray; font-size: 14px; padding: 20px;")
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
                    col = 0
                    row += 1
            
            if len(active_ao_channels) > 1:
                all_paths = [f"{device_name}/ao{ao.replace('AO', '')}" for ao in active_ao_channels]
                c_all = PulseColumn(f"Master: ALL ({len(active_ao_channels)} chs)", all_paths, state_update_callback)
                self.columns.append(c_all)
                grid.addWidget(c_all, row, 0, 1, 2)

        root = QVBoxLayout(self)
        root.addLayout(grid)

    def closeEvent(self, event):
        for col in self.columns:
            col.close()
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
            if c > 7:  
                c = 0; r += 1
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
        for sig in ALL_AI_CHANNELS:
            if self.checkboxes[sig].isChecked(): selected.append(sig)
        for sig in ALL_AO_CHANNELS:
            if self.checkboxes[sig].isChecked(): selected.append(sig)
        if self.checkboxes["DMM"].isChecked(): selected.append("DMM")
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
        
        # --- LEFT PANEL: Export Controls ---
        grid = QGridLayout()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.filename_input = QLineEdit(f"averaged_data_export_{timestamp}")
        self.resample_rate_input = QLineEdit("10") 
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
            cb = QCheckBox(display_name)
            cb.setChecked(True)
            self.export_ch_cbs[sig] = cb
            ch_layout.addWidget(cb, row, col)
            col += 1
            if col > 2: 
                col = 0
                row += 1
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
        
        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.setStyleSheet("font-weight: bold; background-color: #0078D7; color: white; padding: 10px;")
        self.export_btn.clicked.connect(self.process_export)
        left_layout.addWidget(self.export_btn)
        
        # --- RIGHT PANEL: Interactive Preview Graph ---
        preview_top = QHBoxLayout()
        preview_top.addWidget(QLabel("<b>Preview Channel:</b>"))
        self.preview_cb = QComboBox()
        preview_top.addWidget(self.preview_cb)
        preview_top.addStretch()
        right_layout.addLayout(preview_top)

        self.figure = plt.figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        right_layout.addWidget(self.canvas)
        
        self.span = None
        self.tdms_path = self.parent.current_tdms_filepath
        self.time_data = None
        
        # Load TDMS once for fast previewing
        if os.path.exists(self.tdms_path) and not (self.parent.read_thread and self.parent.read_thread.is_alive()):
            try:
                with TdmsFile.read(self.tdms_path) as tdms_file:
                    group = tdms_file["RawData"]
                    self.time_data = group["Time"][:]
                    
                    self.start_time_input.setText(f"{self.time_data[0]:.3f}")
                    self.end_time_input.setText(f"{self.time_data[-1]:.3f}")
                    
                    # Populate dropdown with available channels
                    for sig in self.parent.available_signals:
                        if sig in group:
                            display_name = f"{name_map.get(sig, sig)} ({sig})"
                            self.preview_cb.addItem(display_name, userData=sig)
                            
                self.preview_cb.currentIndexChanged.connect(self.update_preview_plot)
                self.update_preview_plot() # Plot the first one immediately
                    
            except Exception as e:
                self.ax.set_title(f"Could not load preview. Ensure measurement is stopped.")
        else:
            self.ax.set_title("Stop the DAQ measurement to view interactive preview.")
            self.preview_cb.setEnabled(False)
            
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)

    def update_preview_plot(self):
        """Redraws the preview graph instantly when a new channel is selected from the dropdown."""
        if self.time_data is None: return
        
        sig = self.preview_cb.currentData()
        if not sig: return
        
        try:
            with TdmsFile.read(self.tdms_path) as tdms_file:
                group = tdms_file["RawData"]
                if sig in group:
                    y_data = group[sig][:]
                    
                    self.ax.clear()
                    # Decimate heavily for fast visualization
                    step = max(1, len(self.time_data) // 3000)
                    self.ax.plot(self.time_data[::step], y_data[::step], color='#0078D7', alpha=0.8)
                    
                    self.ax.set_title(f"Preview: {self.preview_cb.currentText()} (Click and Drag to select time window)")
                    self.ax.set_xlabel("Time (s)")
                    self.ax.set_ylabel("Amplitude")
                    self.ax.grid(True)
                    
                    # Re-attach the Span Selector
                    self.span = SpanSelector(self.ax, self.on_span_select, 'horizontal', useblit=True)
                    
                    self.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating preview: {e}")

    def on_span_select(self, xmin, xmax):
        """Callback fired when the user draws a box on the interactive graph."""
        self.start_time_input.setText(f"{xmin:.3f}")
        self.end_time_input.setText(f"{xmax:.3f}")
        
    def process_export(self):
        if self.parent.read_thread and self.parent.read_thread.is_alive():
            QMessageBox.warning(self, "Warning", "Please stop the current measurement before exporting.")
            return
            
        tdms_path = self.parent.current_tdms_filepath
        if not tdms_path or not os.path.exists(tdms_path):
            QMessageBox.critical(self, "Error", "No recorded TDMS file found in memory to export.")
            return
            
        try:
            target_rate = float(self.resample_rate_input.text())
            orig_rate = float(self.parent.read_rate_input.text())
            t_start = float(self.start_time_input.text())
            t_end_text = self.end_time_input.text().strip().lower()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid inputs. Please ensure numbers are typed correctly.")
            return

        export_name = f"{self.filename_input.text().strip()}.csv"
        out_csv = os.path.join(self.parent.output_folder, export_name)
        configs = self.parent.get_current_channel_configs()
        cfg_dict = {c["Name"]: c for c in configs}

        try:
            self.export_btn.setText("Reading TDMS...")
            self.export_btn.setEnabled(False)
            QApplication.processEvents()
            
            with TdmsFile.read(tdms_path) as tdms_file:
                group = tdms_file["RawData"]
                time_data = group["Time"][:]
                
                t_end = time_data[-1] if t_end_text == "max" else float(t_end_text)
                mask = (time_data >= t_start) & (time_data <= t_end)
                time_sliced = time_data[mask]
                
                if len(time_sliced) == 0:
                    QMessageBox.warning(self, "Warning", "No data found within the selected time range.")
                    self.export_btn.setText("Export to CSV")
                    self.export_btn.setEnabled(True)
                    return
                
                factor = max(1, int(orig_rate / target_rate))
                valid_len = (len(time_sliced) // factor) * factor
                
                self.export_btn.setText("Resampling Data...")
                QApplication.processEvents()
                
                names, units, raw_names, scales, offsets = ["Time"], ["s"], ["Time"], ["1"], ["0"]
                
                stack = []
                stack.append(time_sliced[:valid_len].reshape(-1, factor).mean(axis=1))
                
                for sig in self.parent.available_signals:
                    if self.export_ch_cbs[sig].isChecked():
                        raw_v = group[sig][:][mask]
                        raw_v_sliced = raw_v[:valid_len].reshape(-1, factor).mean(axis=1)
                        
                        if sig in cfg_dict:
                            cfg = cfg_dict[sig]
                            custom_name = cfg["CustomName"]
                            unit = cfg["Unit"]
                            scale = cfg["Scale"]
                            offset = cfg["Offset"]
                            val = (raw_v_sliced - offset) * scale
                        else:
                            custom_name = "DMM"
                            unit = "V"
                            scale = 1.0
                            offset = 0.0
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
                    f.write(",".join(offsets) + "\n")
                    f.write("\n") 
                    np.savetxt(f, arr, delimiter=",", fmt="%.6g")
                    
            QMessageBox.information(self, "Success", f"Data successfully exported to:\n{out_csv}")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")
            traceback.print_exc()
        finally:
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
        if calc_type == "Frequency":
            self.value_label.setText(f"{value:.2f} Hz")
        else:
            self.value_label.setText(f"{value:.4g} {unit}")

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
            if raw_sig in selected_signals:
                cb.setChecked(True)
            self.checkboxes[raw_sig] = cb
            grid.addWidget(cb, row, col)
            col += 1
            if col > 3:  
                col = 0
                row += 1
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
            
    def update_index(self, index):
        self.title_label.setText(f"Subplot {index + 1}")
        
    def update_mapping(self, mapping):
        self.mapping = mapping

    def on_remove(self):
        self.remove_callback(self)
        
    def get_selected_signals(self):
        return self.selected_signals
        
    def set_selected_signals(self, signals):
        self.selected_signals = signals

class DAQControlApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAQ Control GUI")

        self.dmmbuffer = []
        
        self.write_thread = None
        self.DMMread_thread = None
        self.read_thread = None
        self.processing_thread = None
        self.tdms_writer_thread = None
        
        self.write_stop_flag = threading.Event()
        self.read_stop_flag = threading.Event()
        
        self.write_task = None
        self.write_task_lock = threading.Lock()
        self.history_lock = threading.Lock() 

        self.start_timestamp = None
        self.sample_nr = 0
        self.output_folder = r"\data"
        self.current_tdms_filepath = ""
        
        self.tdms_queue = queue.Queue(maxsize=1000) 
        self.process_queue = queue.Queue(maxsize=100)

        self.available_signals = ["AI0", "AI1", "AI2", "AI3", "AI4", "AI5", "AO0", "AO1", "AO2", "AO3", "DMM"]
        self.master_channel_configs = {sig: self.get_default_channel_config(sig) for sig in ALL_CHANNELS}
        self.active_channel_configs = [] 
        
        self.history_maxlen = 50000 
        self.history_samples = 0
        self.history_time = collections.deque(maxlen=self.history_maxlen)
        self.history_data = {sig: collections.deque(maxlen=self.history_maxlen) for sig in ALL_CHANNELS}
        
        self.current_ao_state = {sig: 0.0 for sig in ALL_AO_CHANNELS}
        
        self.plot_lines = {}
        self.needs_plot_rebuild = True
        self.latest_math_values = {}
        self.active_math_signals = set()

        self.indicator_widgets = []
        
        # FPS Tracker
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
            dev_name = self.device_cb.currentText()
            nidaqmx.system.Device(dev_name).reset_device()
            do_init = nidaqmx.Task()
            do_init.do_channels.add_do_chan(self.get_do_channel(), line_grouping=LineGrouping.CHAN_PER_LINE)
            do_init.write([False, False])
            do_init.close()
        except Exception as e:
            print(f"[WARN] DAQ Hardware not found at startup: {e}. Check 'Simulation Mode' to run anyway.")

    def get_default_channel_config(self, raw_name):
        term = "RSE" if raw_name.startswith("AI") and int(raw_name[2:]) >= 16 else "DIFF"
        return {
            "custom_name": raw_name,
            "term": term,
            "range": "-10 to 10",
            "sensor": "None",
            "scale": "1.0",
            "unit": "V",
            "offset": "0.0"
        }

    def get_write_channel(self):
        return f"{self.device_cb.currentText()}/ao0"
        
    def get_do_channel(self):
        return f"{self.device_cb.currentText()}/port0/line0:1"

    def update_ao_state_from_pulse(self, channels, volts):
        for ch in channels:
            raw_sig = ch.split('/')[-1].upper() 
            if raw_sig in self.current_ao_state:
                self.current_ao_state[raw_sig] = float(volts)

    def launch_analog_out(self):
        device_name = self.device_cb.currentText()
        active_aos = [s for s in self.available_signals if s.startswith("AO")]
        
        if self.ao_window is not None:
            self.ao_window.close()
        self.ao_window = VoltageToggleWindow(device_name, active_aos, self.update_ao_state_from_pulse)
        self.ao_window.show()
        self.ao_window.raise_()

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

        indicator_header = QLabel("<b>Numerical Indicators</b>")
        left_panel.addWidget(indicator_header)
        
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
        self.simulate_checkbox = QCheckBox("Simulation Mode")

        self.fps_label = QLabel("GUI FPS: 0.0")
        self.fps_label.setStyleSheet("color: gray; font-weight: bold;")

        self.folder_display = QLabel()
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.folder_display.setText(f"Output Folder: {self.output_folder if len(self.output_folder)<40 else self.output_folder[-40:]}")
        
        choose_folder_btn = QPushButton("Choose Output Folder")
        choose_folder_btn.clicked.connect(self.select_output_folder)
        
        self.export_ascii_btn = QPushButton("Export ASCII Data (Interactive)")
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
        controls_layout.addWidget(self.simulate_checkbox, 7, 0)
        controls_layout.addWidget(self.fps_label, 7, 1)
        
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

        self.figure = plt.figure(figsize=(10, 10))
        self.canvas = FigureCanvas(self.figure)
        self.axs = []
        self.subplot_widgets = []

        right_panel.addLayout(controls_layout)
        right_panel.addLayout(plot_control_layout) 
        center_panel.addWidget(self.canvas)
        center_panel.addLayout(btn_layout)

        main_layout.addLayout(left_panel, stretch=1)
        main_layout.addLayout(center_panel, stretch=4)
        main_layout.addLayout(right_panel, stretch=1)

    def refresh_devices(self):
        current = self.device_cb.currentText()
        self.device_cb.clear()
        try:
            sys_local = nidaqmx.system.System.local()
            devs = [d.name for d in sys_local.devices]
            if not devs: devs = ["Dev1"]
        except Exception:
            devs = ["Dev1"]
        self.device_cb.addItems(devs)
        if current in devs:
            self.device_cb.setCurrentText(current)

    def update_device_labels(self, new_device):
        for ch in self.channel_ui_configs:
            raw_name = ch['name']
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
            self.master_channel_configs[raw_name] = {
                "custom_name": ch["custom_name_input"].text(),
                "term": ch["term_cb"].currentText(),
                "range": ch["range_cb"].currentText(),
                "sensor": ch["sensor_cb"].currentText(),
                "scale": ch["scale_input"].text(),
                "unit": ch["unit_input"].text(),
                "offset": ch["offset_input"].text()
            }

    def apply_batch_config(self):
        b_term = self.batch_term.currentText()
        b_range = self.batch_range.currentText()
        b_sensor = self.batch_sensor.currentText()

        for ch_ui in self.channel_ui_configs:
            if ch_ui['name'].startswith("AI"):
                idx = int(ch_ui['name'][2:])
                ch_ui['range_cb'].setCurrentText(b_range)
                ch_ui['sensor_cb'].setCurrentText(b_sensor)
                
                if idx >= 16 and b_term == "DIFF":
                    ch_ui['term_cb'].setCurrentText("RSE")
                else:
                    ch_ui['term_cb'].setCurrentText(b_term)

    def setup_config_tab(self):
        main_lay = QVBoxLayout(self.config_tab)
        
        dev_layout = QHBoxLayout()
        dev_layout.addWidget(QLabel("<b>DAQ Device:</b>"))
        
        self.device_cb = QComboBox()
        self.refresh_devices()
        self.device_cb.currentTextChanged.connect(self.update_device_labels)
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
        
        dmm_layout = QHBoxLayout()
        dmm_layout.addWidget(QLabel("<b>Keithley DMM IP:</b>"))
        self.Keithley_DMM_IP = QLineEdit("169.254.169.37")
        dmm_layout.addWidget(self.Keithley_DMM_IP)
        dmm_layout.addStretch()
        main_lay.addLayout(dmm_layout)
        
        bottom_hlay = QHBoxLayout()
        self.measure_offsets_btn = QPushButton("Measure Offsets (1s)")
        self.measure_offsets_btn.clicked.connect(self.measure_ui_offsets)
        self.apply_config_btn = QPushButton("Save Settings & Update UI")
        self.apply_config_btn.setStyleSheet("font-weight: bold; background-color: #28a745; color: white;")
        self.apply_config_btn.clicked.connect(self.apply_config_update)
        
        bottom_hlay.addWidget(self.measure_offsets_btn)
        bottom_hlay.addWidget(self.apply_config_btn)
        main_lay.addLayout(bottom_hlay)
        
        self.rebuild_config_tab()

    def rebuild_config_tab(self):
        while self.config_grid.count():
            item = self.config_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.channel_ui_configs = []
        term_options = ["RSE", "NRSE", "DIFF"]
        term_options_high = ["RSE", "NRSE"] 
        range_options = ["-10 to 10", "-5 to 5", "-2.5 to 2.5", "-0.2 to 0.2"]
        sensor_options = ["None", "Type K"]

        def make_sensor_callback(sensor_cb, scale_input, unit_input, offset_input):
            def callback(index):
                if sensor_cb.currentText() != "None":
                    scale_input.setText("1.0")
                    scale_input.setEnabled(False)
                    offset_input.setText("0.0")
                    offset_input.setEnabled(False)
                    unit_input.setText("°C")
                else:
                    scale_input.setEnabled(True)
                    offset_input.setEnabled(True)
            return callback

        headers = ["Channel", "Custom Name", "Terminal Config", "Voltage Range", "Sensor Type", "Scale Factor", "Unit", "Offset (V)"]
        for col, h in enumerate(headers):
            self.config_grid.addWidget(QLabel(f"<b>{h}</b>"), 0, col)

        row_idx = 1
        dev_prefix = self.device_cb.currentText()
        
        active_ai = [s for s in self.available_signals if s.startswith("AI")]
        if active_ai:
            ai_label = QLabel("<b>Analog Inputs (AI)</b>")
            ai_label.setStyleSheet("color: #0055a4; font-size: 14px; margin-top: 10px; margin-bottom: 5px;")
            self.config_grid.addWidget(ai_label, row_idx, 0, 1, 8)
            row_idx += 1

        for raw_name in self.available_signals:
            if raw_name == "DMM": continue

            if raw_name.startswith("AO") and not any(r['name'].startswith("AO") for r in self.channel_ui_configs):
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setFrameShadow(QFrame.Sunken)
                self.config_grid.addWidget(line, row_idx, 0, 1, 8)
                row_idx += 1
                
                ao_label = QLabel("<b>Analog Outputs (AO)</b>")
                ao_label.setStyleSheet("color: #0055a4; font-size: 14px; margin-top: 5px; margin-bottom: 5px;")
                self.config_grid.addWidget(ao_label, row_idx, 0, 1, 8)
                row_idx += 1

            cfg = self.master_channel_configs[raw_name]
            
            ch_label = QLabel(f"{dev_prefix}/{raw_name.lower()} ({raw_name})")
            custom_name_input = QLineEdit(cfg["custom_name"])
            
            term_cb = QComboBox()
            if raw_name.startswith("AI"):
                idx = int(raw_name[2:])
                if idx >= 16:
                    term_cb.addItems(term_options_high)
                    if cfg["term"] == "DIFF": cfg["term"] = "RSE"
                else:
                    term_cb.addItems(term_options)
            else:
                term_cb.addItems(term_options)
            term_cb.setCurrentText(cfg["term"])
                
            range_cb = QComboBox()
            range_cb.addItems(range_options)
            range_cb.setCurrentText(cfg["range"])
            
            sensor_cb = QComboBox()
            sensor_cb.addItems(sensor_options)
            sensor_cb.setCurrentText(cfg["sensor"])
            
            scale_input = QLineEdit(cfg["scale"])
            unit_input = QLineEdit(cfg["unit"])
            offset_input = QLineEdit(cfg["offset"])

            sensor_cb.currentIndexChanged.connect(make_sensor_callback(sensor_cb, scale_input, unit_input, offset_input))

            if raw_name.startswith("AO"):
                term_cb.setEnabled(False)
                range_cb.setEnabled(False)
                sensor_cb.setEnabled(False)

            self.config_grid.addWidget(ch_label, row_idx, 0)
            self.config_grid.addWidget(custom_name_input, row_idx, 1)
            self.config_grid.addWidget(term_cb, row_idx, 2)
            self.config_grid.addWidget(range_cb, row_idx, 3)
            self.config_grid.addWidget(sensor_cb, row_idx, 4)
            self.config_grid.addWidget(scale_input, row_idx, 5)
            self.config_grid.addWidget(unit_input, row_idx, 6)
            self.config_grid.addWidget(offset_input, row_idx, 7)

            self.channel_ui_configs.append({
                "name": raw_name,
                "ch_label": ch_label,
                "custom_name_input": custom_name_input,
                "term_cb": term_cb,
                "range_cb": range_cb,
                "sensor_cb": sensor_cb,
                "scale_input": scale_input,
                "unit_input": unit_input,
                "offset_input": offset_input
            })
            row_idx += 1

    def apply_config_update(self):
        self.cache_current_ui_configs()
        self.active_channel_configs = self.get_current_channel_configs()
        
        self.latest_math_values = {sig: {"Current (100ms avg)": 0.0, "RMS": 0.0, "Peak-to-Peak": 0.0, "Frequency": 0.0} for sig in self.available_signals}
        
        ind_mapping = []
        sub_mapping = []
        for cfg in self.active_channel_configs:
            raw = cfg['Name']
            custom = cfg['CustomName']
            ind_mapping.append((raw, custom))
            sub_mapping.append((raw, f"{custom} ({raw})"))
            
        if "DMM" in self.available_signals:
            ind_mapping.append(("DMM", "DMM"))
            sub_mapping.append(("DMM", "DMM (DMM)"))
        
        for sub in self.subplot_widgets:
            sub.update_mapping(sub_mapping)
            
        for ind in self.indicator_widgets:
            current_raw = ind.signal_cb.currentData()
            ind.signal_cb.clear()
            for raw, custom in ind_mapping:
                ind.signal_cb.addItem(custom, userData=raw)
            idx = ind.signal_cb.findData(current_raw)
            if idx >= 0: ind.signal_cb.setCurrentIndex(idx)
            elif ind.signal_cb.count() > 0: ind.signal_cb.setCurrentIndex(0)
                
        self.save_config()
        self.needs_plot_rebuild = True
        print("[INFO] Settings saved and UI updated successfully.")

    def flag_plot_rebuild(self):
        self.needs_plot_rebuild = True

    def add_indicator(self, default_signal="AI0", default_type="Current (100ms avg)"):
        ind_mapping = []
        for cfg in self.active_channel_configs:
            ind_mapping.append((cfg['Name'], cfg['CustomName']))
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
        for cfg in self.active_channel_configs:
            sub_mapping.append((cfg['Name'], f"{cfg['CustomName']} ({cfg['Name']})"))
        if "DMM" in self.available_signals: sub_mapping.append(("DMM", "DMM (DMM)"))
        
        widget = SubplotConfigWidget(idx, sub_mapping, self.remove_subplot, self.flag_plot_rebuild)
        if default_signals:
            widget.set_selected_signals(default_signals)
            
        self.plot_scroll_layout.insertWidget(len(self.subplot_widgets), widget)
        self.subplot_widgets.append(widget)
        self.needs_plot_rebuild = True

    def remove_subplot(self, widget):
        self.plot_scroll_layout.removeWidget(widget)
        widget.deleteLater()
        self.subplot_widgets.remove(widget)
        for i, w in enumerate(self.subplot_widgets):
            w.update_index(i)
        self.needs_plot_rebuild = True

    def save_config(self):
        self.cache_current_ui_configs()
        config = {
            "main": {
                "device": self.device_cb.currentText(),
                "write_rate": self.write_rate_input.text(),
                "read_rate": self.read_rate_input.text(),
                "avg_samples": self.average_samples_input.text(),
                "plot_window": self.plot_window_input.text(),
                "threshold": self.threshold_input.text(),
                "dmm_ip": self.Keithley_DMM_IP.text(),
                "simulate": self.simulate_checkbox.isChecked(),
                "output_folder": self.output_folder,
                "available_signals": self.available_signals
            },
            "master_channels": self.master_channel_configs,
            "subplots": [],
            "indicators": []
        }
        for sub in self.subplot_widgets:
            config["subplots"].append(sub.get_selected_signals())
        for ind in self.indicator_widgets:
            config["indicators"].append({
                "signal": ind.signal_cb.currentData(),
                "type": ind.type_cb.currentText()
            })
        try:
            with open("daq_config.json", "w") as f:
                json.dump(config, f, indent=4)
            print("[INFO] Configuration saved to daq_config.json")
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}")

    def load_config(self):
        if not os.path.exists("daq_config.json"): 
            self.add_subplot(["AI0", "AI1"])
            self.add_subplot(["AI2", "AI3"])
            self.add_indicator("AI0", "RMS")
            return False
            
        try:
            with open("daq_config.json", "r") as f: config = json.load(f)
            main_cfg = config.get("main", {})
            
            if "device" in main_cfg:
                idx = self.device_cb.findText(main_cfg["device"])
                if idx >= 0: self.device_cb.setCurrentIndex(idx)
                
            if "write_rate" in main_cfg: self.write_rate_input.setText(main_cfg["write_rate"])
            if "read_rate" in main_cfg: self.read_rate_input.setText(main_cfg["read_rate"])
            if "avg_samples" in main_cfg: self.average_samples_input.setText(main_cfg["avg_samples"])
            if "plot_window" in main_cfg: self.plot_window_input.setText(main_cfg["plot_window"])
            if "threshold" in main_cfg: self.threshold_input.setText(main_cfg["threshold"])
            if "dmm_ip" in main_cfg: self.Keithley_DMM_IP.setText(main_cfg["dmm_ip"])
            if "simulate" in main_cfg: self.simulate_checkbox.setChecked(main_cfg["simulate"])
            if "output_folder" in main_cfg: 
                self.output_folder = main_cfg["output_folder"]
                self.folder_display.setText(f"Output Folder: {self.output_folder if len(self.output_folder)<40 else self.output_folder[-40:]}")

            if "available_signals" in main_cfg:
                self.available_signals = main_cfg["available_signals"]
            
            if "master_channels" in config:
                for k, v in config["master_channels"].items():
                    if k in self.master_channel_configs:
                        self.master_channel_configs[k].update(v)

            self.rebuild_config_tab()
            self.apply_config_update()

            while self.subplot_widgets: self.remove_subplot(self.subplot_widgets[0])
            while self.indicator_widgets: self.remove_indicator(self.indicator_widgets[0])

            for sub_signals in config.get("subplots", []): self.add_subplot(sub_signals)
            for ind_cfg in config.get("indicators", []): self.add_indicator(ind_cfg.get("signal", "AI0"), ind_cfg.get("type", "RMS"))

            print("[INFO] Configuration loaded from daq_config.json")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            return False

    def get_current_channel_configs(self):
        config_map = {
            "RSE": TerminalConfiguration.RSE, "NRSE": TerminalConfiguration.NRSE,
            "DIFF": TerminalConfiguration.DIFF, "PSEUDO_DIFF": TerminalConfiguration.PSEUDO_DIFF
        }
        def parse_range(r_str):
            try:
                parts = r_str.split(" to ")
                return (float(parts[0]), float(parts[1]))
            except: return (-10.0, 10.0)

        dev_prefix = self.device_cb.currentText()
        daq_configs = []
        for ch in self.channel_ui_configs:
            sensor_type = ch['sensor_cb'].currentText()
            try: scale_val = float(ch['scale_input'].text())
            except ValueError: scale_val = 1.0 
            try: offset_val = float(ch['offset_input'].text())
            except ValueError: offset_val = 0.0
                
            if sensor_type != "None":
                scale_val = 1.0
                offset_val = 0.0

            daq_configs.append({
                'Name': ch['name'], 
                'Terminal': f"{dev_prefix}/{ch['name'].lower()}",
                'CustomName': ch['custom_name_input'].text().strip() or ch['name'],
                'Config': config_map[ch['term_cb'].currentText()],
                'Range': parse_range(ch['range_cb'].currentText()),
                'SensorType': sensor_type, 'Scale': scale_val,
                'Unit': ch['unit_input'].text().strip(), 'Offset': offset_val
            })
        return daq_configs

    def measure_ui_offsets(self):
        configs = self.get_current_channel_configs()
        if self.simulate_checkbox.isChecked():
            for ch in self.channel_ui_configs:
                if not ch['name'].startswith("AO") and ch['sensor_cb'].currentText() == "None":
                    ch['offset_input'].setText("0.000000")
            return
            
        active_ais = [c for c in configs if c['Name'].startswith("AI")]
        if not active_ais:
            QMessageBox.information(self, "Info", "No AI channels selected for offset measurement.")
            return
            
        try:
            with nidaqmx.Task() as task:
                for ch in active_ais:
                    if ch['SensorType'] == "Type K":
                        task.ai_channels.add_ai_thrmcpl_chan(ch['Terminal'], thermocouple_type=ThermocoupleType.K, cjc_source=CJCSource.BUILT_IN)
                    else:
                        task.ai_channels.add_ai_voltage_chan(ch['Terminal'], terminal_config=ch['Config'], min_val=ch['Range'][0], max_val=ch['Range'][1])
                        
                task.timing.cfg_samp_clk_timing(rate=1000, sample_mode=AcquisitionType.FINITE, samps_per_chan=1000)
                data = task.read(number_of_samples_per_channel=1000, timeout=3.0)
                
                if len(active_ais) == 1:
                    means = [np.mean(data)]
                else:
                    means = np.mean(data, axis=1)
                
                ai_idx = 0
                for i, ch_ui in enumerate(self.channel_ui_configs):
                    if ch_ui['name'].startswith("AI"):
                        if active_ais[ai_idx]['SensorType'] == "None":
                            ch_ui['offset_input'].setText(f"{means[ai_idx]:.6f}")
                        ai_idx += 1
                        
        except Exception as e: print(f"[ERROR] Failed to measure offsets: {e}")

    # --- DAQ Threads ---
    def start_read(self):
        self.apply_config_update()
        
        if self.DMMread_thread and self.DMMread_thread.is_alive(): return   
        if self.read_thread and self.read_thread.is_alive(): return     
        if self.processing_thread and self.processing_thread.is_alive(): return
        if self.tdms_writer_thread and self.tdms_writer_thread.is_alive(): return
        
        self.read_stop_flag.clear()
        if self.start_timestamp is None: self.start_timestamp = time.time()
            
        with self.history_lock:
            self.history_time.clear()
            for sig in self.available_signals:
                self.history_data[sig].clear()
                
        self.needs_plot_rebuild = True
        
        self.active_math_signals = set()
        for ind in self.indicator_widgets:
            self.active_math_signals.add(ind.signal_cb.currentData())

        while not self.tdms_queue.empty(): self.tdms_queue.get()
        while not self.process_queue.empty(): self.process_queue.get()

        self.tdms_writer_thread = threading.Thread(target=self.tdms_writer_thread_func, name="tdms_writer_thread")
        self.DMMread_thread = threading.Thread(target=self.DMM_read, name="DMMread_thread")
        self.read_thread = threading.Thread(target=self.read_voltages, name="read_voltages")
        self.processing_thread = threading.Thread(target=self.process_data_thread, name="process_data_thread")
        
        self.tdms_writer_thread.start()
        self.DMMread_thread.start()
        time.sleep(0.1)
        self.read_thread.start()
        self.processing_thread.start()
        
        self.gui_timer.start(100)

    def stop_read(self):
        self.read_stop_flag.set()
        self.gui_timer.stop()

    def tdms_writer_thread_func(self):
        filename = self.generate_filename("raw_data")
        self.current_tdms_filepath = str(filename)
        
        active_ai = [s for s in self.available_signals if s.startswith("AI")]
        active_ao = [s for s in self.available_signals if s.startswith("AO")]
        has_dmm = "DMM" in self.available_signals

        try:
            with TdmsWriter(str(filename)) as writer:
                while not self.read_stop_flag.is_set() or not self.tdms_queue.empty():
                    try:
                        chunk = self.tdms_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    
                    time_arr, ai_data, ao_data, dmm_data = chunk
                    channels = [ChannelObject("RawData", "Time", time_arr)]
                    
                    for i, sig in enumerate(active_ai): channels.append(ChannelObject("RawData", sig, ai_data[i]))
                    for i, sig in enumerate(active_ao): channels.append(ChannelObject("RawData", sig, ao_data[i]))
                    if has_dmm: channels.append(ChannelObject("RawData", "DMM", dmm_data.flatten()))
                    
                    writer.write_segment(channels)
            print(f"[INFO] TDMS recording saved: {filename}")
        except Exception as e:
            print(f"[ERROR] TDMS Writer Thread failed: {e}")

    def DMM_read(self):
        has_dmm = "DMM" in self.available_signals
        if not has_dmm: return
        
        if self.simulate_checkbox.isChecked():
            while not self.read_stop_flag.is_set():
                self.dmmbuffer.append(float(np.random.uniform(-0.1, 0.1)))
                time.sleep(0.1)
            return
        inst = None
        try:
            inst = DMM6510readout.write_script_to_Keithley(self.Keithley_DMM_IP.text(), "0.05")
            while not self.read_stop_flag.is_set():
                data = DMM6510readout.read_data(inst)
                self.dmmbuffer.append(float(data))
        except Exception: pass
        finally:
            try:
                if inst is not None: DMM6510readout.stop_instrument(inst)
            except Exception: pass

    def resize_dmmdata(self, length):
        data = np.asarray(self.dmmbuffer)
        n = len(data)
        self.dmmbuffer = self.dmmbuffer[-1:]
        if n == 0: return np.zeros(length)
        if n < length:
            block = length // n
            pattern = np.repeat(data, block)
            remaining = length - len(pattern)
            if remaining > 0: pattern = np.concatenate([pattern, np.repeat(data[-1], remaining)])
            return pattern
        else:
            indices = np.linspace(0, n, length+1, endpoint=True).astype(int)
            output = [data[indices[i]:indices[i+1]].mean() if len(data[indices[i]:indices[i+1]]) > 0 else output[-1] for i in range(length)]
            return np.array(output)

    def read_voltages(self):
        configs = self.active_channel_configs
        
        active_ai_configs = [c for c in configs if c['Name'].startswith("AI")]
        active_ao_configs = [c for c in configs if c['Name'].startswith("AO")]
        
        ai_offsets = np.array([cfg['Offset'] for cfg in active_ai_configs])[:, np.newaxis]
        
        n_ai = len(active_ai_configs)
        n_ao = len(active_ao_configs)
        has_dmm = "DMM" in self.available_signals
        n_total = len(self.available_signals)

        if n_total == 0:
            print("[INFO] No channels selected. Read loop terminating.")
            return

        try:
            self.sample_nr = 0
            try: read_rate = float(self.read_rate_input.text())
            except ValueError: read_rate = 10000.0
            
            samples_per_read = max(1, int(read_rate // 10)) 
            safe_timeout = (samples_per_read / read_rate) + 2.0

            if self.simulate_checkbox.isChecked():
                t_wave = 0
                while not self.read_stop_flag.is_set():
                    t_start = time.time()
                    time_arr = np.linspace(t_wave, t_wave + samples_per_read/read_rate, samples_per_read, endpoint=False)
                    t_wave += samples_per_read/read_rate
                    
                    if n_ai > 0:
                        ai_data = np.random.uniform(-0.1, 0.1, (n_ai, samples_per_read))
                        if n_ai > 0: ai_data[0, :] += np.sin(2 * np.pi * 50 * time_arr) * 2.0  
                        if n_ai > 1: ai_data[1, :] += np.sin(2 * np.pi * 15 * time_arr) * 1.5 
                        
                        for i, cfg in enumerate(active_ai_configs):
                            if cfg['SensorType'] == "Type K":
                                ai_data[i, :] = np.random.uniform(24.5, 25.5, samples_per_read)
                                
                        raw_ai_tdms = ai_data.copy()
                        ai_data = ai_data - ai_offsets 
                    else:
                        ai_data = np.empty((0, samples_per_read))
                        raw_ai_tdms = np.empty((0, samples_per_read))

                    if n_ao > 0:
                        ao_vals = np.array([self.current_ao_state[cfg['Name']] for cfg in active_ao_configs])
                        ao_chunk = np.repeat(ao_vals[:, None], samples_per_read, axis=1)
                    else:
                        ao_chunk = np.empty((0, samples_per_read))

                    if has_dmm:
                        dmm_chunk = self.resize_dmmdata(samples_per_read).reshape(1, -1)
                    else:
                        dmm_chunk = np.empty((0, samples_per_read))
                    
                    global_time = (self.sample_nr + np.arange(samples_per_read)) / read_rate
                    
                    try:
                        self.tdms_queue.put_nowait((global_time, raw_ai_tdms, ao_chunk.copy(), dmm_chunk.copy()))
                    except queue.Full: pass 
                    
                    data_to_process = np.vstack((ai_data, ao_chunk, dmm_chunk)) if n_ai > 0 or n_ao > 0 or has_dmm else np.empty((0, samples_per_read))
                    
                    try:
                        self.process_queue.put_nowait((global_time, data_to_process))
                    except queue.Full: pass 
                    
                    self.sample_nr += samples_per_read
                    
                    elapsed = time.time() - t_start
                    sleep_time = (samples_per_read / read_rate) - elapsed
                    if sleep_time > 0: time.sleep(sleep_time)
                return
            
            ai_data = np.zeros((n_ai, samples_per_read), dtype=np.float64) if n_ai > 0 else np.empty((0, samples_per_read))
            task = nidaqmx.Task() if n_ai > 0 else None
            stream_reader = None

            if task is not None:
                for ch in active_ai_configs:
                    if ch['SensorType'] == "Type K":
                        task.ai_channels.add_ai_thrmcpl_chan(ch['Terminal'], thermocouple_type=ThermocoupleType.K, cjc_source=CJCSource.BUILT_IN)
                    else:
                        task.ai_channels.add_ai_voltage_chan(ch['Terminal'], terminal_config=ch['Config'], min_val=ch['Range'][0], max_val=ch['Range'][1])

                task.timing.cfg_samp_clk_timing(rate=read_rate, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=int(read_rate * 10))
                stream_reader = stream_readers.AnalogMultiChannelReader(task.in_stream)
                task.start()

            while not self.read_stop_flag.is_set():
                try:
                    if task is not None:
                        stream_reader.read_many_sample(data=ai_data, number_of_samples_per_channel=samples_per_read, timeout=safe_timeout)
                        raw_ai_tdms = ai_data.copy()
                        ai_data_processed = ai_data - ai_offsets
                    else:
                        time.sleep(samples_per_read / read_rate)
                        ai_data_processed = ai_data
                        raw_ai_tdms = ai_data

                except Exception: continue

                if n_ao > 0:
                    ao_vals = np.array([self.current_ao_state[cfg['Name']] for cfg in active_ao_configs])
                    ao_chunk = np.repeat(ao_vals[:, None], samples_per_read, axis=1)
                else:
                    ao_chunk = np.empty((0, samples_per_read))

                if has_dmm:
                    dmm_chunk = self.resize_dmmdata(samples_per_read).reshape(1, -1)
                else:
                    dmm_chunk = np.empty((0, samples_per_read))

                global_time = (self.sample_nr + np.arange(samples_per_read)) / read_rate
                
                try:
                    self.tdms_queue.put_nowait((global_time, raw_ai_tdms, ao_chunk.copy(), dmm_chunk.copy()))
                except queue.Full: pass
                
                data_to_process = np.vstack((ai_data_processed, ao_chunk, dmm_chunk))
                try:
                    self.process_queue.put_nowait((global_time, data_to_process))
                except queue.Full: pass

                self.sample_nr += samples_per_read

        except Exception as e:
            print(f"[ERROR] read_voltages crashed: {e}")
        finally:
            self.read_stop_flag.set()
            if 'task' in locals() and task is not None:
                try:
                    task.stop()
                    task.close()
                except Exception: pass

    def process_data_thread(self):
        try: rate = float(self.read_rate_input.text())
        except ValueError: rate = 10000.0
        try: average_samples = max(1, int(self.average_samples_input.text()))
        except ValueError: average_samples = 100
        
        configs = self.active_channel_configs
        scales = {cfg["Name"]: cfg["Scale"] for cfg in configs}
        scales["DMM"] = 1.0 
        
        num_signals = len(self.available_signals)
        if num_signals == 0: return

        math_samps = int(rate * 0.5)
        math_buffer = np.zeros((num_signals, math_samps), dtype=np.float64)

        accum_data = []
        accum_t = []
        accum_len = 0

        while not self.read_stop_flag.is_set():
            try:
                t_chunk, data_chunk = self.process_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            n_new = data_chunk.shape[1]
            
            if n_new >= math_samps:
                math_buffer = data_chunk[:, -math_samps:]
            else:
                math_buffer = np.roll(math_buffer, -n_new, axis=1)
                math_buffer[:, -n_new:] = data_chunk

            samples_100ms = int(rate * 0.1)
            active_maths = self.active_math_signals
            
            for i, sig in enumerate(self.available_signals):
                if sig not in active_maths:
                    continue 
                
                scale = scales.get(sig, 1.0)
                sig_data = math_buffer[i, :] * scale
                if len(sig_data) == 0: continue
                
                cur_avg = np.mean(sig_data[-samples_100ms:]) if len(sig_data) > samples_100ms else np.mean(sig_data)
                rms = np.sqrt(np.mean(np.square(sig_data)))
                p2p = np.max(sig_data) - np.min(sig_data)
                freq = 0.0
                centered = sig_data - np.mean(sig_data)
                crossings = np.where((centered[:-1] < 0) & (centered[1:] >= 0))[0]
                if len(crossings) > 1:
                    dt = (crossings[-1] - crossings[0]) / rate
                    if dt > 0: freq = (len(crossings) - 1) / dt
                    
                self.latest_math_values[sig]["Current (100ms avg)"] = cur_avg
                self.latest_math_values[sig]["RMS"] = rms
                self.latest_math_values[sig]["Peak-to-Peak"] = p2p
                self.latest_math_values[sig]["Frequency"] = freq

            accum_data.append(data_chunk)
            accum_t.append(t_chunk)
            accum_len += n_new

            if accum_len >= average_samples:
                big_data = np.concatenate(accum_data, axis=1)
                big_t = np.concatenate(accum_t)
                
                n_points = accum_len // average_samples
                valid_len = n_points * average_samples
                
                block_data = big_data[:, :valid_len]
                block_t = big_t[:valid_len]
                
                reshaped_data = block_data.reshape((num_signals, n_points, average_samples))
                avg_data = np.mean(reshaped_data, axis=2)
                
                reshaped_t = block_t.reshape((n_points, average_samples))
                avg_t = np.mean(reshaped_t, axis=1)

                with self.history_lock:
                    self.history_time.extend(avg_t)
                    for i, sig in enumerate(self.available_signals):
                        self.history_data[sig].extend(avg_data[i, :] * scales.get(sig, 1.0))

                rem_data = big_data[:, valid_len:]
                rem_t = big_t[valid_len:]
                
                accum_data = [rem_data] if rem_data.shape[1] > 0 else []
                accum_t = [rem_t] if rem_t.shape[0] > 0 else []
                accum_len = rem_data.shape[1]

    def update_gui(self):
        # Calculate Frame Rate (FPS)
        now = time.perf_counter()
        dt = now - self.last_update_time
        self.last_update_time = now
        if dt > 0:
            self.fps_queue.append(1.0 / dt)
        if len(self.fps_queue) > 0:
            avg_fps = sum(self.fps_queue) / len(self.fps_queue)
            self.fps_label.setText(f"GUI FPS: {avg_fps:.1f}")

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

        try: window_s = float(self.plot_window_input.text())
        except ValueError: window_s = 10.0
        try:
            rate = float(self.read_rate_input.text())
            avg_samps = max(1, int(self.average_samples_input.text()))
            eff_plot_rate = rate / avg_samps
        except ValueError: eff_plot_rate = 100.0 
        
        num_points = int(window_s * eff_plot_rate) if window_s > 0 else 0

        required_signals = set()
        for widget in self.subplot_widgets:
            for sig in widget.get_selected_signals():
                required_signals.add(sig)

        with self.history_lock:
            if len(self.history_time) == 0: return
            
            if num_points > 0 and len(self.history_time) > num_points:
                t_plot = list(collections.deque(self.history_time, maxlen=num_points))
                y_plots = {sig: list(collections.deque(self.history_data[sig], maxlen=num_points)) for sig in required_signals}
            else:
                t_plot = list(self.history_time)
                y_plots = {sig: list(self.history_data[sig]) for sig in required_signals}

        # FAST VISUAL DECIMATION FILTER FOR MATPLOTLIB
        MAX_VISUAL_POINTS = 1500 
        step = max(1, len(t_plot) // MAX_VISUAL_POINTS)
        
        t_arr = np.array(t_plot)[::step]

        if self.needs_plot_rebuild:
            self.figure.clear()
            self.axs = []
            self.plot_lines = {}
            n = len(self.subplot_widgets)
            
            if n > 0:
                for i in range(n):
                    ax = self.figure.add_subplot(n, 1, i + 1)
                    self.axs.append(ax)
                    self.plot_lines[i] = {}
                    
                    selected_raw_signals = self.subplot_widgets[i].get_selected_signals()
                    plot_units = set()
                    
                    for raw_sig in selected_raw_signals:
                        if raw_sig not in self.available_signals: continue
                        unit = units.get(raw_sig, "V")
                        c_name = custom_names.get(raw_sig, raw_sig)
                        plot_units.add(unit)
                        
                        line, = ax.plot([], [], label=f"{c_name} [{unit}]")
                        self.plot_lines[i][raw_sig] = line
                    
                    if plot_units: 
                        ax.set_ylabel(f"Value [{', '.join(sorted(list(plot_units)))}]")
                    else: 
                        ax.set_ylabel("Value")
                    
                    if selected_raw_signals: 
                        ax.legend(loc='upper right')
                    ax.grid('on')
                    if i == n - 1: 
                        ax.set_xlabel("Time (s)")
                    
            self.figure.subplots_adjust(hspace=0.4)
            self.needs_plot_rebuild = False

        for i, ax in enumerate(self.axs):
            selected_raw_signals = self.subplot_widgets[i].get_selected_signals()
            for raw_sig in selected_raw_signals:
                if raw_sig in self.plot_lines[i] and raw_sig in y_plots:
                    y_arr = np.array(y_plots[raw_sig])[::step]
                    self.plot_lines[i][raw_sig].set_data(t_arr, y_arr)
            
            if len(t_arr) > 1:
                ax.set_xlim(left=t_arr[0], right=t_arr[-1])
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            elif len(t_arr) == 1:
                ax.set_xlim(left=max(0, t_arr[0]-1), right=t_arr[0]+1)

        self.canvas.draw_idle()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.folder_display.setText(f"Output Folder: {self.output_folder if len(self.output_folder)<40 else self.output_folder[-40:]}")

    def generate_filename(self, base_name=""):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        default_name = f"{base_name.strip()}_{timestamp}.tdms" if base_name.strip() else f"{timestamp}.tdms"
        folder_path = Path(self.output_folder) if self.output_folder else Path.cwd()
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path / default_name

    def generate_profile(self, write_rate):
        return [] 

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
                except Exception: pass
            
            write_channel = self.get_write_channel()
            self.write_task = nidaqmx.Task()
            self.write_task.ao_channels.add_ao_voltage_chan(write_channel)
            self.write_task.timing.cfg_samp_clk_timing(write_rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(voltages))
            try: self.write_task.write(voltages, auto_start=True)
            except nidaqmx.errors.DaqError:
                try: self.write_task.close()
                except Exception: pass
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
                except Exception: pass
                self.write_task = None
        try: self.zero_ao_output()
        except Exception: pass

    def zero_ao_output(self):
        if self.simulate_checkbox.isChecked():
            self.last_output_voltage = 0.0
            return
        try:
            with nidaqmx.Task() as zero_task:
                zero_task.ao_channels.add_ao_voltage_chan(self.get_write_channel())
                zero_task.write(0.0); time.sleep(0.01); zero_task.write(0.0)
            self.last_output_voltage = 0.0
        except Exception: pass

    def reset_all(self):
        self.write_stop_flag.set()
        self.read_stop_flag.set()
        self.gui_timer.stop()

        if self.write_thread and self.write_thread.is_alive(): self.write_thread.join(timeout=3)
        if self.read_thread and self.read_thread.is_alive(): self.read_thread.join(timeout=3)
        if self.processing_thread and self.processing_thread.is_alive(): self.processing_thread.join(timeout=3)
        if self.tdms_writer_thread and self.tdms_writer_thread.is_alive(): self.tdms_writer_thread.join(timeout=3)

        with self.write_task_lock:
            if self.write_task is not None:
                try: self.write_task.stop(); self.write_task.close()
                except Exception: pass
                self.write_task = None

        self.zero_ao_output()
        
        with self.history_lock:
            self.history_time.clear()
            for sig in self.available_signals:
                self.history_data[sig].clear()

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
        dialog = ExportDialog(self)
        dialog.exec_()

    def closeEvent(self, event):
        print("[INFO] Exiting application. Saving config, stopping tasks and zeroing AO0...")
        self.save_config()
        self.write_stop_flag.set()
        self.read_stop_flag.set()
        self.gui_timer.stop()
        
        if hasattr(self, 'ao_window') and self.ao_window is not None:
            self.ao_window.close() 
        
        if self.write_thread and self.write_thread.is_alive(): self.write_thread.join(timeout=3)
        if self.read_thread and self.read_thread.is_alive(): self.read_thread.join(timeout=3)
        if self.processing_thread and self.processing_thread.is_alive(): self.processing_thread.join(timeout=3)
        if self.tdms_writer_thread and self.tdms_writer_thread.is_alive(): self.tdms_writer_thread.join(timeout=3)
        
        with self.write_task_lock:
            if self.write_task is not None:
                try: self.write_task.stop(); self.write_task.close()
                except Exception: pass
        self.zero_ao_output()
        event.accept()

    def exit_application(self):
        self.close() 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DAQControlApp()
    gui.show()
    sys.exit(app.exec_())   