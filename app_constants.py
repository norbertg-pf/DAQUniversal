from hardware_profiles import all_ai_channels, all_ao_channels

ALL_AI_CHANNELS = all_ai_channels()
ALL_AO_CHANNELS = all_ao_channels()
ALL_MATH_CHANNELS = [f"MATH{i}" for i in range(4)]
ALL_CHANNELS = ALL_AI_CHANNELS + ALL_AO_CHANNELS + ALL_MATH_CHANNELS + ["DMM"]

PLOT_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
]
