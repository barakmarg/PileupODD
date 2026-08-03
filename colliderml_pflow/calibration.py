"""Detector calibration and voxelisation lookup tables.

Both tables are keyed on the ColliderML ``detector`` id:

===  ====================
id   subsystem
===  ====================
 9   ECal endcap (neg)
10   ECal barrel
11   ECal endcap (pos)
12   HCal endcap (neg)
13   HCal barrel
14   HCal endcap (pos)
===  ====================
"""

import polars as pl

# Sampling-fraction correction applied to raw ``total_energy`` to recover the
# incident energy in GeV. Derived from a geometric analysis of the ColliderML
# calo_hits sample; ECal and HCal have different sampling fractions, and the
# endcaps differ slightly from the barrel.
CALIBRATION = pl.DataFrame({
    "detector": [10, 9, 11, 13, 12, 14],
    "system_label": [
        "Ecal Barrel",
        "Ecal Endcap (Neg)", "Ecal Endcap (Pos)",
        "Hcal Barrel",
        "Hcal Endcap (Neg)", "Hcal Endcap (Pos)",
    ],
    "calib_factor": [37.5, 38.7, 38.7, 45.0, 46.9, 46.9],
})

# Voxel edge length in mm used to merge neighbouring cells before clustering.
# The ECal endcaps (9, 11) are finely segmented and use a 25 mm grid; the ECal
# barrel and the whole HCal use 60 mm. Hits are therefore binned on a
# per-subsystem grid rather than a single global one.
voxel_config = pl.DataFrame({
    "detector": [9, 10, 11, 12, 13, 14],
    "v_size": [25.0, 60.0, 25.0, 60.0, 60.0, 60.0],
}).with_columns([
    pl.col("detector").cast(pl.UInt8),
    pl.col("v_size").cast(pl.Float32),
])
