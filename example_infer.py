"""Example: run the packaged TinyViT-21M Phase-2 best checkpoint on one image.

From repo root (d:\\changing):
  python packages/tinyvit21m_phase2_best/example_infer.py --image path/to.jpg
  python packages/tinyvit21m_phase2_best/example_infer.py --image path/to.jpg --question "What is this?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer_utils import generate_answer, load_phase2_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer with packaged Phase-2 best weights")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--smol-dir", type=Path, default=ROOT / "weights" / "smol")
    parser.add_argument("--tinyvit-dir", type=Path, default=ROOT / "weights" / "tinyvit")
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    model, tokenizer, device, ckpt = load_phase2_model(
        PKG,
        smol_dir=args.smol_dir,
        tinyvit_dir=args.tinyvit_dir,
    )
    print(f"checkpoint: {ckpt}")
    print(f"device:     {device}")
    print(f"task:       {args.question or 'Describe the image.'}")

    text = generate_answer(
        model,
        tokenizer,
        args.image,
        question=args.question,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"answer:     {text}")


if __name__ == "__main__":
    main()
