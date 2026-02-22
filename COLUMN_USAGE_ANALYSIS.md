# Column Usage Analysis - For Optimized Loading

## MINIMUM COLUMNS TO DOWNLOAD

### PARTICLES - 17 columns (instead of downloading all)
```python
particles_cols = [
    'event_id',
    'particle_id',
    'vertex_primary',
    'pdg_id',
    'energy',
    'px',
    'py',
    'pz',
    'charge',
    'parent_id',
]
```
**Load code:**
```python
particles = pl.read_parquet(file_path, columns=particles_cols)
```

---

### TRACKS - All columns are used
```python
# No selection needed - load everything
tracks = pl.read_parquet(file_path)
```
**However, if you know specific track features aren't needed:**
```python
tracks_cols = [
    'event_id',
    'majority_particle_id',
    # + any other track features your model uses
]
```

---

### CALO_HITS - 11 columns (instead of downloading all)
```python
calo_hits_cols = [
    'event_id',
    'detector',
    'total_energy',
    'x',
    'y',
    'z',
    'contrib_particle_ids',
    'contrib_energies'

]
```
**Load code:**
```python
calo_hits = pl.read_parquet(file_path, columns=calo_hits_cols)
```

---

## EXPECTED DOWNLOAD TIME REDUCTION

- **PARTICLES**: If original has 50+ columns, downloading only 17 = ~66% reduction
- **CALO_HITS**: If original has 30+ columns, downloading only 11 = ~63% reduction
- **TRACKS**: Depends on columns you actually use

**Total estimated download time reduction: 40-60%**



