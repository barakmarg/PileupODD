import sys
sys.path.insert(0, '/storage/agrp/barakma/PileupODD')
from primary.create_trainning_dataset import run_preprocessing_pipeline

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python submit_preprocess.py <range_start> <range_end> <event_name>")
        sys.exit(1)
    range_start = int(sys.argv[1])
    range_end = int(sys.argv[2])
    event_name = sys.argv[3]
    run_preprocessing_pipeline(range(range_start, range_end), event_name=event_name)