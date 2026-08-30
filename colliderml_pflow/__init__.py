"""Particle-flow training-dataset construction for the ColliderML ODD sample.

Turns the CERN *ColliderML-Release-1* Geant4 simulation of the Open Data
Detector into four flat, ML-ready Parquet tables per shard, in three modes:

``hard_scatter``
    Targets built from hard-scatter particles only (``vertex_primary == 1``).

``all_vertices``
    Targets built from every vertex, so pileup particles are reconstruction
    targets too.

``overlay``
    Synthetic pileup: PU0 hard-scatter events with ``Poisson(mu)`` pileup-only
    events overlaid, including a time-of-flight read-out cut.

Typical use is through the CLI (``python -m colliderml_pflow preprocess
--config ...``); see the README. The library entry points are
:func:`colliderml_pflow.runner.run_preprocessing` for a whole run and
:func:`colliderml_pflow.pipeline.preprocess_events` for a single batch of
in-memory frames.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
