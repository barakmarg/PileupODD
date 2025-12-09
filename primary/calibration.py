import polars as pl

# 1. Define the Calibration Mapping Table
# Based on the geometric analysis of ColliderML calo_hits
CALIBRATION = pl.DataFrame({
    "detector": [10, 9, 11, 13, 12, 14],
    "system_label": [
        "Ecal Barrel", 
        "Ecal Endcap (Neg)", "Ecal Endcap (Pos)", 
        "Hcal Barrel", 
        "Hcal Endcap (Neg)", "Hcal Endcap (Pos)"
    ],
    "calib_factor": [37.5, 38.7, 38.7, 45.0, 46.9, 46.9]
})