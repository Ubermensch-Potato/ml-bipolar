"""Download the TabPFN-3 checkpoint from the gated HuggingFace repo.

TabPFN-3 weights live in a GATED repo (Prior-Labs/tabpfn_3). Before running:
  1. Log in at https://huggingface.co and accept the terms on
     https://huggingface.co/Prior-Labs/tabpfn_3  (Access repository)
  2. Create a READ token at https://huggingface.co/settings/tokens
  3. Provide it via the HF_TOKEN env var (or `hf auth login`).

Usage:
  HF_TOKEN=<hf_read_token> python download_tabpfn.py
  python download_tabpfn.py --file tabpfn-v3-classifier-v3_default.ckpt

The checkpoint (~213 MB) is saved to ml_code/tabpfn_models/ ; run_tabpfn.py and
run_models.py then load it locally via model_path (no token needed at run time).
"""
import os
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "Prior-Labs/tabpfn_3"
DEFAULT_FILE = "tabpfn-v3-classifier-v3_default.ckpt"


def main():
    ap = argparse.ArgumentParser(description="Download the TabPFN-3 checkpoint.")
    ap.add_argument("--file", default=DEFAULT_FILE, help="checkpoint filename in the repo")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--out", default=os.path.join(HERE, "tabpfn_models"))
    a = ap.parse_args()

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print("[warn] HF_TOKEN not set. This is a gated repo — set a HuggingFace read token:\n"
              "       HF_TOKEN=<token> python download_tabpfn.py\n"
              "       (and accept terms at https://huggingface.co/%s)" % a.repo)

    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=a.repo, filename=a.file, local_dir=a.out)
    except Exception as e:
        raise SystemExit(
            f"[error] download failed: {type(e).__name__}: {e}\n"
            f"  - accept terms:  https://huggingface.co/{a.repo}\n"
            f"  - set token:     export HF_TOKEN=<hf read token>  (from huggingface.co/settings/tokens)")
    size_mb = os.path.getsize(path) / 1e6
    print(f"saved: {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
