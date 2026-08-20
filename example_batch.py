"""Minimal batch example over a folder of images (caption mode).

  python packages/tinyvit21m_phase2_best/example_batch.py --image-dir path/to/images --out captions.txt
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

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    images = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in EXTS)
    images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images under {args.image_dir}")

    model, tok, device, _ = load_phase2_model(PKG)
    lines = []
    for i, path in enumerate(images, 1):
        ans = generate_answer(model, tok, path, question=args.question, device=device)
        line = f"{path.name}\t{ans}"
        print(f"[{i}/{len(images)}] {line}")
        lines.append(line)

    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
