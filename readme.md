# Install dependecies

```
python -m venv .venv
.venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install PyQt5 nidaqmx numpy matplotlib nptdms pyvisa google-api-python-client google-auth-httplib2 google-auth-oauthlib scipy
```

## Hardware profile detection

The DAQ channel layout is profile-driven and selected automatically from the chosen DAQ device in the **Channel Config** tab.

- The app detects device type (for example `USB-6453` or `PXIe-6381`) from NI device info.
- The active AI/AO channel set is updated automatically for that selected device.

If you add support for a new NI card, add its profile and aliases in `hardware_profiles.py`.

## Keithley 6510 (single-channel DMM) integration

The Keithley 6510 is integrated as an **external DMM channel** (`DMM`) and is available as a dedicated virtual device entry (`Keithley 6510 (DMM)`) in the DAQ dropdown.

Current behavior:

- NI devices are discovered from NI-DAQmx and shown in the DAQ device dropdown.
- The Keithley value is read separately over VISA (`pyvisa`) and appended as the `DMM` signal.
- `DMM` can be selected together with NI AI/AO channels for simultaneous acquisition.
- Since Keithley data is ~100 Hz, samples are duplicated (zero-order hold) to match the higher NI sample stream when building each DAQ chunk.
- In the channel selector, you can enable `DMM` as one channel alongside NI signals.
- In **Channel Config**, the `DMM` channel supports AI-like processing fields: custom name, scale, unit, offset, and LPF settings.

How to use:

1. Connect the instrument over LAN/USB/GPIB and ensure VISA can see it.
2. In the app, configure the DMM connection settings in the **External Devices** section.
3. Open channel selection and enable the `DMM` signal.

The Keithley entry is virtual (not NI-DAQmx-enumerated hardware), so NI physical devices and the Keithley mode are intentionally handled through different acquisition paths.

## Build standalone Windows `.exe` (PyInstaller)

A one-file PyInstaller spec is included as `DAQ_ui.spec`.

1. Install runtime + build dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install pyinstaller pyqtgraph
```

2. Build a single executable and place it in `build`:

```bash
mkdir -p build
python -m PyInstaller --noconfirm --clean --distpath build --workpath build-temp DAQ_ui.spec
```

Result:

- `build/DAQUniversalReader.exe` (standalone one-file app)

Notes:

- Build on Windows to produce a Windows `.exe`.
- NI-DAQ/driver communication still requires compatible NI drivers/hardware on the target machine.
- If `pyinstaller` is not recognized as a command, always use `python -m PyInstaller ...` as shown above.
