"""Run the all-vertices chunked research preprocessing pipeline.

Wraps run_preprocessing_pipeline_all_vertices_chunked from
create_trainning_dataset_pileup_research.py.

Edit FILE_INDICES below to choose exactly which HF repo file indices to
process, and EVENT_NAME / CHUNK_SIZE as needed.
"""
import sys

sys.path.insert(0, '/storage/agrp/barakma/PileupODD')

from create_trainning_dataset_pileup_research import (
    run_preprocessing_pipeline_all_vertices_chunked,
)

# --- configuration ---------------------------------------------------------
# Explicit list of file indices to process (0..999).
# validation files for training TT bar pileup 200
# /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260615-T233516/ckpts/epoch=083-val_loss=11.27779.ckpt
# https://www.comet.com/barakmarg/odd-pflow-reco/49679a388ebe4a1caa123cec9096d9cf?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step
FILE_INDICES = [

    687, 215, 364, 380, 167, 374, 855, 651, 465, 954, 947, 104, 888, 978, 
    876, 82, 297, 792, 234, 678, 47, 72, 722, 197, 592, 371, 883, 907, 
    634, 644, 850, 301, 566, 81, 388, 128, 550, 471, 748, 45, 827, 271, 
    619, 353, 868, 368, 100, 390, 95, 990, 345, 782, 221, 160, 964, 349, 
    983, 715, 164, 826, 778, 7, 891, 829, 285, 604, 460, 226, 430, 986, 
    719, 666, 734, 204, 575, 28, 617, 518, 239, 224, 96, 31, 33, 433, 
    605, 90, 559, 914, 759, 693, 105, 755, 143, 229, 251, 282, 760, 26, 
    115, 655

]

EVENT_NAME = "ttbar_pu200"
CHUNK_SIZE = 100
# ---------------------------------------------------------------------------


def main() -> None:
    # Optional positional args slice into FILE_INDICES (a range over the list,
    # NOT over file indices). E.g. `python run_research_preprocess.py 0 2`
    # processes FILE_INDICES[0:2] -> indices 687, 215.
    # With no args, the whole FILE_INDICES list is processed.
    start = int(sys.argv[1]) if len(sys.argv) > 1 else None
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None
    selected = FILE_INDICES[start:end]

    print(
        f"Processing {len(selected)} file indices "
        f"(FILE_INDICES[{start}:{end}]): {selected}"
    )
    run_preprocessing_pipeline_all_vertices_chunked(
        r=selected,
        event_name=EVENT_NAME,
        chunk_size=CHUNK_SIZE,
    )


if __name__ == "__main__":
    main()
