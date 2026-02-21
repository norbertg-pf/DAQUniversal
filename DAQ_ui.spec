# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for a single-file Windows executable."""

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hiddenimports = [
    "PyQt5.sip",
    "pyqtgraph",
    "pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt5",
    "nidaqmx",
    "nidaqmx.system",
    "nptdms",
    "pyvisa",
    "scipy",
    "numpy",
] + collect_submodules("pyqtgraph")

a = Analysis(
    ["DAQ.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("DMM6510.tsp", "."),
        ("daq_config.json", "."),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="DAQUniversalReader",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
)
