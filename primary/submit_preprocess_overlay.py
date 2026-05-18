
import sys
# Use the local CLUEstering source clone (has TBB .so) instead of the pip-installed
# wheel in ~/.local site-packages (which lacks TBB).
sys.path.insert(0, '/storage/agrp/barakma/PileupODD')
from primary.create_training_dataset_pileup_overlay import run_preprocessing_pipeline

if __name__ == "__main__":

    
    run_preprocessing_pipeline(range(0, 1), num_of_events=100, clue_backend='gpu cuda', chunk_size=100)