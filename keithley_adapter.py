"""Adapter for reading Keithley 6510/DAQ6510 measurements.

Prefers the optional ``keithley_daq6510`` package when installed, and falls
back to a direct PyVISA SCPI session.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass

import pyvisa


@dataclass
class KeithleyAdapter:
    ip_address: str
    sample_interval_s: float = 0.05

    def __post_init__(self):
        self._backend_name = "pyvisa"
        self._dmm = None
        self._resource = None

    def connect(self):
        package = self._load_keithley_package()
        if package is not None:
            instance = self._try_create_library_client(package)
            if instance is not None:
                self._backend_name = "keithley_daq6510"
                self._dmm = instance
                self._configure_library_client(instance)
                return

        rm = pyvisa.ResourceManager()
        self._resource = rm.open_resource(f"TCPIP0::{self.ip_address}::inst0::INSTR")
        self._resource.timeout = 5000
        self._resource.write("*RST")
        self._resource.write("*CLS")
        self._resource.write('SENS:FUNC "VOLT:DC"')
        self._resource.write(f"SENS:VOLT:DC:NPLC {max(self.sample_interval_s * 10, 0.01):.2f}")

    def read(self) -> float:
        if self._dmm is not None:
            return float(self._read_from_library_client(self._dmm))
        if self._resource is None:
            raise RuntimeError("Keithley instrument is not connected")
        return float(self._resource.query("READ?").strip())

    def close(self):
        if self._dmm is not None:
            for method_name in ("close", "disconnect", "shutdown"):
                method = getattr(self._dmm, method_name, None)
                if callable(method):
                    method()
                    break
            self._dmm = None
        if self._resource is not None:
            try:
                self._resource.write("ABORt")
                self._resource.write("*CLS")
            finally:
                self._resource.close()
                self._resource = None

    def _load_keithley_package(self):
        try:
            return importlib.import_module("keithley_daq6510")
        except ImportError:
            return None

    def _try_create_library_client(self, package):
        class_candidates = (
            "KeithleyDAQ6510",
            "DAQ6510",
            "Keithley6510",
        )
        endpoint = f"TCPIP0::{self.ip_address}::inst0::INSTR"

        for class_name in class_candidates:
            cls = getattr(package, class_name, None)
            if cls is None:
                continue
            for argument in (self.ip_address, endpoint):
                try:
                    return cls(argument)
                except TypeError:
                    continue
                except Exception:
                    break

        connect_fn = getattr(package, "connect", None)
        if callable(connect_fn):
            for argument in (self.ip_address, endpoint):
                try:
                    return connect_fn(argument)
                except TypeError:
                    continue

        return None

    def _configure_library_client(self, instance):
        for method_name in ("set_sample_interval", "set_sample_time", "configure"):
            method = getattr(instance, method_name, None)
            if callable(method):
                try:
                    sig = inspect.signature(method)
                    if len(sig.parameters) == 1:
                        method(self.sample_interval_s)
                    else:
                        method(sample_interval=self.sample_interval_s)
                except Exception:
                    pass
                break

    def _read_from_library_client(self, instance):
        method_candidates = (
            "read",
            "fetch",
            "measure",
            "measure_voltage",
            "read_voltage",
            "read_voltage_dc",
            "get_voltage",
        )
        for method_name in method_candidates:
            method = getattr(instance, method_name, None)
            if callable(method):
                value = method()
                if isinstance(value, (list, tuple)) and value:
                    return value[0]
                return value
        raise RuntimeError("keithley_daq6510 client has no known read method")
