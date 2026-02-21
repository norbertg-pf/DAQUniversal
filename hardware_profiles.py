"""Hardware profile definitions and auto-detection for supported NI DAQ cards."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    ai_count: int
    ao_count: int
    default_enabled_ai: List[int]
    default_enabled_ao: List[int]

    @property
    def ai_channels(self) -> List[str]:
        return [f"AI{i}" for i in range(self.ai_count)]

    @property
    def ao_channels(self) -> List[str]:
        return [f"AO{i}" for i in range(self.ao_count)]

    @property
    def default_enabled_signals(self) -> List[str]:
        enabled_ai = [f"AI{i}" for i in self.default_enabled_ai if 0 <= i < self.ai_count]
        enabled_ao = [f"AO{i}" for i in self.default_enabled_ao if 0 <= i < self.ao_count]
        return enabled_ai + enabled_ao + ["DMM"]


USB_6453_PROFILE = HardwareProfile(
    name="USB-6453",
    ai_count=32,
    ao_count=4,
    default_enabled_ai=[0, 1, 2, 3, 4, 5],
    default_enabled_ao=[0, 1, 2, 3],
)

PXIE_6381_PROFILE = HardwareProfile(
    name="PXIe-6381",
    ai_count=32,
    ao_count=4,
    default_enabled_ai=[0, 1, 2, 3, 4, 5],
    default_enabled_ao=[0, 1, 2, 3],
)

PROFILE_REGISTRY: Dict[str, HardwareProfile] = {
    USB_6453_PROFILE.name: USB_6453_PROFILE,
    PXIE_6381_PROFILE.name: PXIE_6381_PROFILE,
}

DEFAULT_PROFILE_NAME = USB_6453_PROFILE.name

# lower-case substring aliases -> canonical profile name
DEVICE_PROFILE_ALIASES: Dict[str, str] = {
    "usb-6453": USB_6453_PROFILE.name,
    "ni usb-6453": USB_6453_PROFILE.name,
    "pxie-6381": PXIE_6381_PROFILE.name,
    "ni pxie-6381": PXIE_6381_PROFILE.name,
}


def get_profile(profile_name: str) -> HardwareProfile:
    if profile_name not in PROFILE_REGISTRY:
        available = ", ".join(sorted(PROFILE_REGISTRY))
        raise ValueError(f"Unknown hardware profile '{profile_name}'. Available: {available}")
    return PROFILE_REGISTRY[profile_name]


def detect_profile_name(device_product_type: str, device_name: str = "") -> str:
    haystack = f"{device_product_type} {device_name}".strip().lower()
    for alias, profile_name in DEVICE_PROFILE_ALIASES.items():
        if alias in haystack:
            return profile_name
    return DEFAULT_PROFILE_NAME


def all_ai_channels() -> List[str]:
    max_ai = max(profile.ai_count for profile in PROFILE_REGISTRY.values())
    return [f"AI{i}" for i in range(max_ai)]


def all_ao_channels() -> List[str]:
    max_ao = max(profile.ao_count for profile in PROFILE_REGISTRY.values())
    return [f"AO{i}" for i in range(max_ao)]
