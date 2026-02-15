import polars as pl

PDG_ID_TO_NAME = {
    # --- Quarks ---
    1: "d",
    -1: "anti-d",
    2: "u",
    -2: "anti-u",
    3: "s",
    -3: "anti-s",
    4: "c",
    -4: "anti-c",
    5: "b",
    -5: "anti-b",
    6: "t",
    -6: "anti-t",

    # --- Leptons ---
    11: "e-",
    -11: "e+",
    12: "νe",
    -12: "anti-νe",
    13: "μ-",
    -13: "μ+",
    14: "νμ",
    -14: "anti-νμ",
    15: "τ-",
    -15: "τ+",
    16: "ντ",
    -16: "anti-ντ",

    # --- Gauge Bosons ---
    21: "g",
    22: "γ",
    23: "Z0",
    24: "W+",
    -24: "W-",
    25: "H",  # Higgs

    # --- Light Mesons ---
    111: "π0",
    211: "π+",
    -211: "π-",
    221: "η",
    331: "η′",
    113: "ρ0",
    213: "ρ+",
    -213: "ρ-",
    223: "ω",
    333: "φ",
    
    # --- Strange Mesons ---
    130: "K0L",
    310: "K0S",
    311: "K0",
    -311: "anti-K0",
    321: "K+",
    -321: "K-",
    313: "K*0",
    -313: "anti-K*0",
    323: "K*+",
    -323: "K*-",

    # --- Charmed Mesons ---
    411: "D+",
    -411: "D-",
    421: "D0",
    -421: "anti-D0",
    431: "Ds+",
    -431: "Ds-",
    413: "D*+",
    -413: "D*-",
    423: "D*0",
    -423: "anti-D*0",
    433: "Ds*+",
    -433: "Ds*-",

    # --- Charmonium ---
    441: "ηc",
    443: "J/ψ",

    # --- Bottom Mesons ---
    511: "B0",
    -511: "anti-B0",
    521: "B+",
    -521: "B-",
    531: "Bs0",
    -531: "anti-Bs0",
    541: "Bc+",
    -541: "Bc-",
    513: "B*0",
    -513: "anti-B*0",
    523: "B*+",
    -523: "B*-",

    # --- Bottomonium ---
    553: "Υ(1S)",

    # --- Light Baryons ---
    2112: "n",
    -2112: "anti-n",
    2212: "p",
    -2212: "anti-p",
    
    # Delta Baryons
    1114: "Δ-",
    -1114: "anti-Δ+",
    2114: "Δ0",
    -2114: "anti-Δ0",
    2214: "Δ+",
    -2214: "anti-Δ-",
    2224: "Δ++",
    -2224: "anti-Δ--",

    # Strange Baryons
    3122: "Λ",
    -3122: "anti-Λ",
    3222: "Σ+",
    -3222: "anti-Σ-",
    3212: "Σ0",
    -3212: "anti-Σ0",
    3112: "Σ-",
    -3112: "anti-Σ+",
    3312: "Ξ-",
    -3312: "anti-Ξ+",
    3322: "Ξ0",
    -3322: "anti-Ξ0",
    3334: "Ω-",
    -3334: "anti-Ω+",

    # --- Charmed Baryons ---
    4122: "Λc+",
    -4122: "anti-Λc-",
    4222: "Σc++",
    -4222: "anti-Σc--",
    4212: "Σc+",
    -4212: "anti-Σc-",
    4112: "Σc0",
    -4112: "anti-Σc0",
    4232: "Ξc+",
    -4232: "anti-Ξc-",
    4132: "Ξc0",
    -4132: "anti-Ξc0",
    4322: "Ξ'c+",
    -4322: "anti-Ξ'c-",
    4312: "Ξ'c0",
    -4312: "anti-Ξ'c0",
    4332: "Ωc0",
    -4332: "anti-Ωc0",

    # --- Bottom Baryons ---
    5122: "Λb0",
    -5122: "anti-Λb0",
    5112: "Σb-",
    -5112: "anti-Σb+",
    5212: "Σb0",
    -5212: "anti-Σb0",
    5222: "Σb+",
    -5222: "anti-Σb-",
    5132: "Ξb-",
    -5132: "anti-Ξb+",
    5232: "Ξb0",
    -5232: "anti-Ξb0",
    5332: "Ωb-",
    -5332: "anti-Ωb+",

    # --- Nuclei & Ions (PDG 10LZZZAAAI) ---
    1000010020: "deuteron",
    -1000010020: "anti-deuteron",
    1000010030: "triton",
    -1000010030: "anti-triton",
    1000020030: "He3",
    -1000020030: "anti-He3",
    1000020040: "He4",  # Alpha
    -1000020040: "anti-He4",
    1000030060: "Li6",
    1000030070: "Li7",
    1000040090: "Be9",
    1000050100: "B10",
    1000050110: "B11",
    1000060120: "C12",
    1000060130: "C13",
    1000070140: "N14",
    1000080160: "O16",
    1000090190: "F19",
    1000120240: "Mg24",
    1000120250: "Mg25",
    1000120260: "Mg26",
    1000130260: "Al26",
    1000130270: "Al27",
    1000140280: "Si28",
    1000140290: "Si29",
    1000140300: "Si30",
    
    # --- Special / Resonance ---
    -20213: "a1(1260)-",
    20213: "a1(1260)+",
    9000221: "f0(980)",
    -999: "residual/unknown"
}
PDG_ID_TO_NAME = {str(k): v for k, v in PDG_ID_TO_NAME.items()}

STABLE_PDG_IDS = [
    # --- Leptons (reach calorimeter) ---
    11, -11,    # e-, e+
    12, -12,    # ve, anti-ve
    13, -13,    # mu-, mu+
    14, -14,    # vmu, anti-vmu
    16, -16,    # vtau, anti-vtau

    # --- Gauge Bosons ---
    22,         # gamma

    # --- Long-Lived Mesons ---
    130,        # K0L (c*tau ~ 15m)
    211, -211,  # pi+, pi- (c*tau ~ 7.8m)
    321, -321,  # K+, K- (c*tau ~ 3.7m)

    # --- Stable Baryons ---
    2112, -2112, # n, anti-n
    2212, -2212, # p, anti-p

    # --- Nuclei & Ions ---
    1000010020, -1000010020, # Deuteron
    1000010030, -1000010030, # Triton
    1000020030, -1000020030, # He3
    1000020040, -1000020040, # He4
    1000030060, # Li6
    1000030070, # Li7
    1000040090, # Be9
    1000050100, # B10
    1000050110, # B11
    1000060120, # C12
    1000060130, # C13
    1000070140, # N14
    1000080160, # O16
    1000090190, # F19
    1000120240, # Mg24
    1000120250, # Mg25
    1000120260, # Mg26
    1000130260, # Al26
    1000130270, # Al27
    1000140280, # Si28
    1000140290, # Si29
    1000140300  # Si30
]
# Reverse mapping: particle name to PDG ID (assuming unique names)
particle_name_to_id = {v: int(k) for k, v in PDG_ID_TO_NAME.items()}

stable_pdg_ids_df = pl.DataFrame({"pdg_id": STABLE_PDG_IDS})

particles_decaying_immediately = [
    # --- Quarks --- 
    # Lifetime: N/A (Hadronize immediately, ~10^-24 s timescale)
    1, -1,   # d, anti-d
    2, -2,   # u, anti-u
    3, -3,   # s, anti-s
    4, -4,   # c, anti-c
    5, -5,   # b, anti-b
    6, -6,   # t, anti-t (Decays before hadronization: ~5e-25 s)

    # --- Leptons ---
    # e, mu, nu are stable or long-lived enough to reach Calo.
    15, -15, # tau +/- : ~2.90e-13 s (decay length c*tau ~ 87 µm)

    # --- Gauge Bosons ---
    # Gamma (22) is stable.
    21,      # Gluon: N/A (Hadronizes immediately)
    23,      # Z0: ~2.6e-25 s
    24, -24, # W+/-: ~3.0e-25 s
    25,      # Higgs: ~1.6e-22 s

    # --- Light Mesons ---
    # Pi+/- (211) are long-lived (~2.6e-8 s).
    111,     # Pi0: ~8.52e-17 s (Electromagnetic decay)
    221,     # eta: ~5.02e-19 s
    331,     # eta': ~2.0e-21 s
    113,     # rho0: ~4.5e-24 s (Strong decay)
    213, -213, # rho+/-: ~4.5e-24 s
    223,     # omega: ~7.75e-23 s
    333,     # phi: ~1.55e-22 s

    # --- Strange Mesons ---
    # K+/- (321) and K0L (130) are long-lived (~1.2e-8 s and ~5.1e-8 s).
    310,     # K0S: ~8.95e-11 s (c*tau ~ 2.7 cm; decays in tracker)
    311, -311, # K0: N/A (Quantum superposition, mixes to K0S/K0L immediately)
    313, -313, # K*0: ~1.3e-23 s
    323, -323, # K*+/-: ~1.3e-23 s

    # --- Charmed Mesons ---
    # All decay via Weak force, but short lived (c*tau < 300 µm)
    411, -411, # D+/-: ~1.04e-12 s
    421, -421, # D0: ~4.10e-13 s
    431, -431, # Ds+/-: ~5.00e-13 s
    413, -413, # D*+/-: ~8e-21 s (Strong/EM decay)
    423, -423, # D*0: ~infinity (width < 2.1 MeV, very fast decay)
    433, -433, # Ds*+/-: (Decays electromagnetically)

    # --- Charmonium ---
    441,     # eta_c: ~2e-23 s
    443,     # J/psi: ~7.1e-21 s

    # --- Bottom Mesons ---
    # Decay via Weak force (c*tau ~ 450 µm)
    511, -511, # B0: ~1.52e-12 s
    521, -521, # B+: ~1.63e-12 s
    531, -531, # Bs0: ~1.51e-12 s
    541, -541, # Bc+: ~5.1e-13 s
    513, -513, # B*0: (Electromagnetic decay, fast)
    523, -523, # B*+: (Electromagnetic decay, fast)

    # --- Bottomonium ---
    553,     # Upsilon(1S): ~1.2e-20 s

    # --- Light Baryons ---
    # p (2212) and n (2112) are stable/long-lived.
    # Delta Baryons decay via Strong force (~10^-24 s)
    1114, -1114, 2114, -2114, 2214, -2214, 2224, -2224, # Delta lifetimes ~5.6e-24 s

    # --- Strange Baryons (Hyperons) ---
    # Note: Lambda/Sigma/Xi/Omega have c*tau of a few cm. 
    # They decay in the Vertex/Tracker volume, NOT the Calorimeter.
    3122, -3122, # Lambda: ~2.63e-10 s
    3222, -3222, # Sigma+: ~8.02e-11 s
    3212, -3212, # Sigma0: ~7.4e-20 s (Electromagnetic decay to Lambda gamma)
    3112, -3112, # Sigma-: ~1.48e-10 s
    3312, -3312, # Xi-: ~1.64e-10 s
    3322, -3322, # Xi0: ~2.90e-10 s
    3334, -3334, # Omega-: ~8.21e-11 s

    # --- Charmed Baryons ---
    # Lifetimes generally ~10^-13 s
    4122, -4122, # Lambda_c+: ~2.00e-13 s
    4222, -4222, # Sigma_c++: ~1.7e-22 s (Strong decay)
    4212, -4212, # Sigma_c+: ~1.7e-22 s
    4112, -4112, # Sigma_c0: ~1.7e-22 s
    4232, -4232, # Xi_c+: ~4.42e-13 s
    4132, -4132, # Xi_c0: ~1.12e-13 s
    4322, -4322, # Xi'_c: (Electromagnetic decay)
    4312, -4312, # Xi'_c0: (Electromagnetic decay)
    4332, -4332, # Omega_c0: ~6.9e-14 s

    # --- Bottom Baryons ---
    # Lifetimes generally ~10^-12 s
    5122, -5122, # Lambda_b0: ~1.47e-12 s
    5112, -5112, # Sigma_b-: (Strong decay, fast)
    5212, -5212, # Sigma_b0: (Strong decay, fast)
    5222, -5222, # Sigma_b+: (Strong decay, fast)
    5132, -5132, # Xi_b-: ~1.57e-12 s
    5232, -5232, # Xi_b0: ~1.48e-12 s
    5332, -5332, # Omega_b-: ~1.64e-12 s

    # --- Special / Resonance ---
    -20213, 20213, # a1(1260): ~1.6e-24 s
    9000221,       # f0(980): ~1e-23 s
]
unstable_pdg_ids_df = pl.DataFrame({"pdg_id": particles_decaying_immediately})