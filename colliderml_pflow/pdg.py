"""PDG-id tables used by the target-particle definition.

Only the "decays immediately" set is needed by the pipeline. It is consumed by
:func:`colliderml_pflow.preprocessing.set_target_particles_maskv4`, which uses
it to reject particles that could never leave a detector signature of their
own -- quarks, gluons, and short-lived resonances -- so that the target is
built from their decay products instead.

The list is reproduced verbatim from ``primary/pdg_mappings.py`` on ``master``.
That module also defines a ``STABLE_PDG_IDS`` table and a name<->id mapping;
neither is reachable from the pipeline, so neither is carried over here.
"""

import polars as pl


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
