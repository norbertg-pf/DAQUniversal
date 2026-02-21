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

## Build standalone Windows `.exe` (PyInstaller)

A one-file PyInstaller spec is included as `DAQ_ui.spec`.

1. Install runtime + build dependencies:

```bash
pip install pyinstaller pyqtgraph
```

2. Build a single executable and place it in `library/binaries`:

```bash
mkdir -p library/binaries
pyinstaller --noconfirm --clean --distpath library/binaries --workpath build DAQ_ui.spec
```

Result:

- `library/binaries/DAQUniversalReader.exe` (standalone one-file app)

Notes:

- Build on Windows to produce a Windows `.exe`.
- NI-DAQ/driver communication still requires compatible NI drivers/hardware on the target machine.
