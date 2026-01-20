from primary.preprocessing import backtrack_to_target
import polars as pl

def link_target_to_truth(particles: pl.DataFrame) -> pl.DataFrame:
    target_p = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'is_target_particle'])
        .explode( 'particle_id','is_target_particle')
        .filter(pl.col('is_target_particle') )
        .select(['event_id', 'particle_id'])
    )
    truth_p = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'is_parent_missing'])
        .explode('particle_id','is_parent_missing')
        .filter(pl.col('is_parent_missing') )
        .select(['event_id', 'particle_id'])
    )
    mappings = backtrack_to_target(particles=particles, src_df=target_p, target_df=truth_p).rename({'particle_id_src':'target_particle_id','particle_id_target':'truth_particle_id'})
    return (
        particles.lazy()
        .select(['event_id', 'particle_id', 'pt', 'eta'])
        .explode(['particle_id', 'pt', 'eta']) 
        .join(mappings.lazy(),
            left_on=['event_id', 'particle_id'],
                right_on=['event_id', 'target_particle_id']
            , how='inner')
        .group_by(['event_id', 'truth_particle_id'])
        .agg([pl.col('pt').sum().alias('total_target_pt')])
        .join(
            particles.lazy()
            .select(['event_id', 'particle_id', 'pt'])
            .explode(['particle_id', 'pt']) 
            .rename({'pt':'truth_pt', 'particle_id':'truth_particle_id'}),
            on=['event_id', 'truth_particle_id'],
            how='inner'
        )
    ).collect()