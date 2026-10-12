"""Entry point for ViSO / FlatS1 training on ViVQA-S1 (new file; train.py is untouched).
    python train_viso.py --config-file configs/viso_vivqav2.yaml --opts MODEL.DEVICE cpu LOSS.NAME soft_ce ...
The explicit imports below register the new modules in the existing registries (no __init__ was edited)."""
import argparse
import warnings

warnings.filterwarnings("ignore")

import data_utils.vocabs.s1_vocab  # noqa: F401,E402  (META_VOCAB: S1Vocab)
import data_utils.datasets.s1_dataset  # noqa: F401,E402  (META_DATASET: S1FeatureDataset)
import data_utils.datasets.s1_live_dataset  # noqa: F401,E402  (META_DATASET: S1LiveDataset, ViSO-FT)
import models.viso  # noqa: F401,E402  (META_ARCHITECTURE: ViSO, FlatS1)
import tasks.typed_decision_task  # noqa: F401,E402  (META_TASK: TypedDecisionTask)
from builders.task_builder import build_task  # noqa: E402
from configs.utils import get_config  # noqa: E402
from utils.logging_utils import setup_logger  # noqa: E402

logger = setup_logger()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--opts", nargs="*", default=[], help="KEY VALUE pairs overriding the yaml (yacs syntax)")
    return ap.parse_args(argv)


def load_config(config_file, opts):
    cfg = get_config(config_file)
    cfg.merge_from_list(opts)
    return cfg


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_file, args.opts)
    task = build_task(config)
    task.start()
    task.get_predictions()
    logger.info("Task done.")
