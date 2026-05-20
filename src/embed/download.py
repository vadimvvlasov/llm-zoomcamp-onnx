import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

ONNX_CANDIDATES = [
    "onnx/model.onnx",
    "onnx/encoder_model.onnx",
    "model.onnx",
]


def download(repo, dest="models"):
    # Strip a leading dest-prefix if the user accidentally passed a local path
    # e.g. "models/Xenova/bge-base-en-v1.5" → "Xenova/bge-base-en-v1.5"
    repo = (
        str(Path(repo).relative_to(dest))
        if str(repo).startswith(str(dest) + "/")
        else repo
    )

    dest = Path(dest) / repo
    dest.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(repo_id=repo)
    onnx_file = next((c for c in ONNX_CANDIDATES if c in files), None)
    if not onnx_file:
        raise FileNotFoundError(f"No ONNX model found in {repo}")

    for remote, local in [
        ("tokenizer.json", "tokenizer.json"),
        (onnx_file, "model.onnx"),
    ]:
        src = hf_hub_download(repo_id=repo, filename=remote)
        dst = dest / local
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  saved {dst}")
        else:
            print(f"  exists {dst}")

    onnx_ext = onnx_file + "_data"
    if onnx_ext in files:
        src = hf_hub_download(repo_id=repo, filename=onnx_ext)
        dst = dest / "model.onnx_data"
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  saved {dst}")
        else:
            print(f"  exists {dst}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download an ONNX model from Hugging Face Hub",
        epilog=(
            "examples:\n"
            "  python -m src.embed.download\n"
            "  python -m src.embed.download sentence-transformers/all-MiniLM-L6-v2\n"
            "  python -m src.embed.download Xenova/all-MiniLM-L6-v2 --dest /tmp/models\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "repo",
        nargs="?",
        default="Xenova/all-MiniLM-L6-v2",
        help="HF repo id, e.g. Xenova/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--dest", default="models", help="Destination directory (default: models)"
    )
    args = parser.parse_args()

    download(args.repo, args.dest)
