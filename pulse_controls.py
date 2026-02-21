import time

import nidaqmx
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


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

