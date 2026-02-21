import os
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from nptdms import TdmsFile
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app_constants import ALL_AI_CHANNELS, ALL_AO_CHANNELS, ALL_CHANNELS, ALL_MATH_CHANNELS

try:
    import googleapiclient.discovery  # noqa: F401
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False


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
        self.setWindowTitle("Export ASCII Data")
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
