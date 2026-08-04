"""Label-invariant summaries of the cluster-dependent output tables.

Calorimeter clustering is stochastic. CLUE hands out cluster ids in discovery
order, and its CUDA backend reduces in a nondeterministic order, so two runs
over identical input produce different cluster *labels* and slightly different
cluster boundaries. Comparing ``calo_clusters`` or ``target_particles_deps``
row by row therefore cannot work -- not between this branch and ``master``, and
not even between two runs of ``master`` itself.

What *is* comparable are quantities that do not depend on how clusters happen
to be labelled or ordered:

- counts of physically meaningful things (clusters, contributing vertices,
  target particles that received energy);
- total energies, summed over clusters;
- energy grouped by *physical* identifiers -- ``vertex_primary`` (a real vertex)
  and ``particle_idx`` (stable, since it comes from the deterministic
  ``target_particles`` table) -- rather than by ``cluster_idx``, which is a
  per-run label.

Measured on a two-event ``ttbar_pu200`` fixture with the CUDA backend, across
two ``master`` runs and two branch runs:

===========================  ==============  ==============  ==============
quantity                     master/master   branch/branch   master/branch
===========================  ==============  ==============  ==============
n_clusters, n_vtx,           exact           exact           exact
n_particles_with_deps
energies                     <= 0.004%       <= 0.002%       <= 0.004%
n_hits                       <= 0.0003%      <= 0.0003%      <= 0.002%
n_links                      <= 0.008%       <= 0.024%       <= 0.058%
===========================  ==============  ==============  ==============

So branch-vs-master sits in the same band as master's own run-to-run spread.

The tolerances applied are those numbers with a factor
:data:`TOLERANCE_HEADROOM` of headroom. Overlay needs it: with ~200 pileup
interactions overlaid per event, cluster boundaries move more than in the
``all_vertices`` fixture measured above, and subset sums such as ``E_hcal``
fluctuate with them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl

#: Headroom over the spread measured on the ``all_vertices`` fixture below.
#: Overlay is noisier than that fixture -- ~200 pileup interactions land on each
#: event, so cluster boundaries move more and subset sums such as ``E_hcal``
#: fluctuate correspondingly -- and 1x the all_vertices numbers turned out to be
#: marginal there.
TOLERANCE_HEADROOM = 3

#: Quantities that count things. Looser bound: cluster-boundary jitter moves a
#: few hits, and hence a few incidence links, between clusters.
COUNT_RTOL = TOLERANCE_HEADROOM * 5e-3      # 1.5%

#: Quantities that sum energy. Tighter: totals are conserved, and the residual
#: is float32 summation order plus whatever the moving cluster boundaries carry.
ENERGY_RTOL = TOLERANCE_HEADROOM * 1e-3     # 0.3%

COUNT_KEYS = ("n_clusters", "n_hits", "n_vtx", "n_links", "n_particles_with_deps")
ENERGY_KEYS = ("E_total", "E_hcal", "E_vtx_total", "E_deps")


def summarise(out_dir: Path) -> Dict[str, float]:
    """Reduce one run's output to label-invariant per-event quantities.

    Args:
        out_dir: directory holding ``calo_clusters.parquet`` and
            ``target_particles_deps.parquet``.

    Returns:
        ``{"e<event>.<quantity>": value}``.
    """
    clusters = pl.read_parquet(out_dir / "calo_clusters.parquet").sort("event_id")
    deps = pl.read_parquet(out_dir / "target_particles_deps.parquet").sort("event_id")

    result: Dict[str, float] = {}
    for row in range(clusters.height):
        prefix = f"e{row}."
        result[prefix + "n_clusters"] = float(len(clusters["cluster_id"][row]))
        result[prefix + "E_total"] = float(sum(clusters["total_cluster_energy"][row].to_list()))
        result[prefix + "E_hcal"] = float(sum(clusters["hcal_energy"][row].to_list()))
        result[prefix + "n_hits"] = float(sum(clusters["number_of_hits"][row].to_list()))

        # Energy per physical vertex, summed across clusters. Independent of how
        # the energy was partitioned into clusters.
        per_vertex: Dict[int, float] = {}
        for indices, energies in zip(clusters["vertex_primary_indices"][row],
                                     clusters["vertex_primary_energies"][row]):
            for vertex, energy in zip(indices.to_list(), energies.to_list()):
                per_vertex[vertex] = per_vertex.get(vertex, 0.0) + energy
        result[prefix + "n_vtx"] = float(len(per_vertex))
        result[prefix + "E_vtx_total"] = float(sum(per_vertex.values()))

        # Deposits keyed on particle_idx, which is stable because it is derived
        # from the deterministic target_particles table.
        particle_idx = np.array(deps["particle_idx"][row].to_list())
        energy = np.array(deps["total_energy_deps_in_cluster"][row].to_list())
        result[prefix + "n_links"] = float(len(particle_idx))
        result[prefix + "E_deps"] = float(energy.sum())
        result[prefix + "n_particles_with_deps"] = float(len(set(particle_idx.tolist())))
    return result


def relative_deviation(left: Dict[str, float], right: Dict[str, float]) -> Dict[str, float]:
    """Per-quantity relative deviation, keyed the same way as :func:`summarise`."""
    assert set(left) == set(right), (
        f"summaries cover different quantities: {sorted(set(left) ^ set(right))}"
    )
    out = {}
    for key, value in left.items():
        out[key] = abs(right[key] - value) / abs(value) if value else abs(right[key])
    return out


def tolerance_for(key: str) -> float:
    """Allowed relative deviation for a quantity produced by :func:`summarise`."""
    name = key.split(".", 1)[1]
    if name in COUNT_KEYS:
        return COUNT_RTOL
    if name in ENERGY_KEYS:
        return ENERGY_RTOL
    raise AssertionError(f"no tolerance defined for invariant {name!r}")


def deposits_per_particle(out_dir: Path, event_row: int) -> Dict[int, float]:
    """Total deposited energy per target particle for one event.

    Keyed on ``particle_idx``, so it is comparable across runs even though the
    cluster each deposit landed in is not.
    """
    deps = pl.read_parquet(out_dir / "target_particles_deps.parquet").sort("event_id")
    particle_idx = deps["particle_idx"][event_row].to_list()
    energy = deps["total_energy_deps_in_cluster"][event_row].to_list()
    totals: Dict[int, float] = {}
    for pid, e in zip(particle_idx, energy):
        totals[pid] = totals.get(pid, 0.0) + e
    return totals
