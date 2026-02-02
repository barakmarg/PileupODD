from typing import Dict
import polars as pl
import yaml # type: ignore

from sklearn.model_selection import train_test_split
from primary.preprocessing import add_eta_and_phi_and_pt, add_eta_and_phi_and_pt, add_ms_cluster_labels, add_ms_cluster_labels, \
     add_orphan_mask, add_created_inside_calo_mask, add_particle_have_track_mask, set_target_particles_maskv3, get_particles_id_parent_of_inside_calo_particles_maskv3, \
    add_eta_and_phi_and_pt, add_ms_cluster_labels, backtrack_to_target, cluster_purity, calculate_extrapolated_features_polars
from primary.calibration import CALIBRATION


def create_truth_records(particles: pl.DataFrame, tracks: pl.DataFrame, calo_hits: pl.DataFrame,

                         num_of_events: int=-1,  truth_eta_cut: float=3.0, truth_pt_cut: float=1.0, target_pt_cut: float=0.3, clusters_cutoff: float=0.1):
    if num_of_events >= 0:
        particles = particles.filter(pl.col("event_id") <num_of_events)
        tracks = tracks.filter(pl.col("event_id") <num_of_events)
        calo_hits = calo_hits.filter(pl.col("event_id") <num_of_events)

    # Cast to Float32
    particles = particles.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.List(pl.Float64)).cast(pl.List(pl.Float32))
    ])
    tracks = tracks.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.List(pl.Float64)).cast(pl.List(pl.Float32))
    ])
    calo_hits = calo_hits.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.List(pl.Float64)).cast(pl.List(pl.Float32))
    ])

    particles = add_orphan_mask(particles)
    particles = add_created_inside_calo_mask(particles)
    particles = add_particle_have_track_mask(particles, tracks)
    particles = add_eta_and_phi_and_pt(particles)
    particles = get_particles_id_parent_of_inside_calo_particles_maskv3(particles, calo_hits)
    particles = set_target_particles_maskv3(particles, truth_eta_cut=truth_eta_cut, truth_pt_cut=truth_pt_cut, target_pt_cut=target_pt_cut)

    particles_truth = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'is_truth_particle', 'pdg_id',
              'energy', 'eta', 'phi', 'px', 'py', 'pz', 'pt',
              'charge','mass', 'has_track'])
        .explode( 'particle_id', 'is_truth_particle', 'pdg_id',
              'energy', 'eta', 'phi', 'px', 'py', 'pz', 'pt',
              'charge','mass', 'has_track')
        .filter(pl.col('is_truth_particle'))
        .sort('event_id')
        .with_row_index("global_order")
        .sort('global_order')
        .drop('is_truth_particle', 'global_order')
        .group_by('event_id', maintain_order=True)
        .agg('*')
        .collect(streaming=True)
    )
    return {
        'truth_particles': particles_truth,
    }

def run_truth_pipeline(r=None, event_name: str="ttbar_pu0", ):
    from huggingface_hub import HfFileSystem
    import polars as pl
    import tqdm
    import gc
    fs = HfFileSystem()
    if r is not None:
        number_of_files = r
    for i in tqdm.tqdm(number_of_files):
        file_path = f"datasets/CERN/ColliderML-Release-1/data/{event_name}_particles/train-{i:05d}-of-00100.parquet"
        print(f"Processing file: {file_path}")
        if not fs.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_path = f"datasets/CERN/ColliderML-Release-1/data/{event_name}_particles/train-{i:05d}-of-00100.parquet"
        with fs.open(file_path, "rb") as f:
            particles = pl.read_parquet(f)


        file_path = f"datasets/CERN/ColliderML-Release-1/data/{event_name}_calo_hits/train-{i:05d}-of-00100.parquet"
        with fs.open(file_path, "rb") as f:
            calo_hits = pl.read_parquet(f)

        file_path = f"datasets/CERN/ColliderML-Release-1/data/{event_name}_tracks/train-{i:05d}-of-00100.parquet"
        with fs.open(file_path, "rb") as f:
            tracks = pl.read_parquet(f)

        preprocessed_data = create_truth_records(particles=particles, tracks=tracks,
                                                  calo_hits=calo_hits, num_of_events=-1, 
                                                  truth_pt_cut=1, truth_eta_cut=3.0, target_pt_cut=0.3, clusters_cutoff=0.23)
        
        # write preprocessed data to local disk as parquets
        file_path_data = f"/storage/agrp/barakma/PileupODD/data/{event_name}"
        from pathlib import Path
        Path(file_path_data).mkdir(parents=True, exist_ok=True)
        for key, df in preprocessed_data.items():
            df.write_parquet(f"{file_path_data}/{key}-{i:05d}.parquet")
        
        # Free memory
        del particles, tracks, calo_hits, preprocessed_data
        gc.collect()