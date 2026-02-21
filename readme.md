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
