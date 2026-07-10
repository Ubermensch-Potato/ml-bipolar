"""Project paths — everything is resolved relative to the ml_code/ root so the
runners work no matter which sub-package they live in."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ml_code/
CONFIG = os.path.join(ROOT, "config.yaml")
OUTPUTS = os.path.join(ROOT, "outputs")
TABPFN_DIR = os.path.join(ROOT, "tabpfn_models")
TABPFN_CKPT = os.path.join(TABPFN_DIR, "tabpfn-v3-classifier-v3_default.ckpt")

os.makedirs(OUTPUTS, exist_ok=True)


def resolve_data_path(cfg_path):
    """Locate the dataset: $BIPOLAR_DATA_PATH wins, else config's path (relative -> ml_code/)."""
    path = os.environ.get("BIPOLAR_DATA_PATH") or cfg_path
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"dataset not found: {path}\n"
            f"Set data.path in {CONFIG}, or export BIPOLAR_DATA_PATH=/abs/path/to/bipolar_dataset.xlsx"
        )
    return path
