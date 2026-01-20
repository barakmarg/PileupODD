import sys
sys.path.insert(0, '/storage/agrp/barakma/PileupODD')
from primary.create_trainning_dataset import run_preprocessing_pipeline

if __name__ == "__main__":
    run_preprocessing_pipeline(range(50,55), event_name='ttbar_pu0')