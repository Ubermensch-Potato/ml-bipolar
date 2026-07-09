"""Project paths — everything is resolved relative to the ml_code/ root so the
runners work no matter which sub-package they live in."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ml_code/
CONFIG = os.path.join(ROOT, "config.yaml")
OUTPUTS = os.path.join(ROOT, "outputs")
TABPFN_DIR = os.path.join(ROOT, "tabpfn_models")
TABPFN_CKPT = os.path.join(TABPFN_DIR, "tabpfn-v3-classifier-v3_default.ckpt")

os.makedirs(OUTPUTS, exist_ok=True)
