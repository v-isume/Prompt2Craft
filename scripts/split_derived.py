#!/usr/bin/env python3
"""Regenerate the hand-derived images of the ImageNet-single dataset.

``rebuild_dataset.py`` can copy every ``imagenet`` file verbatim, but it lists
the ``derived`` files (``n<synset>_<id>_<k>.<ext>``) as "not reproduced" because
they were produced by a manual crop step. This tool encodes that manual step so
the derived files can be recreated from their synset origin.

Currently one origin needs splitting: ``n04190052_340.JPEG`` is a side-by-side
photo of the same bookshelf before/after painting. The curated dataset keeps the
two halves as separate images:

    n04190052_340.JPEG (500x332)
      |-- left half  -> n04190052_340_1.JPEG   (unpainted, light wood)
      `-- right half -> n04190052_340_2.JPEG   (painted, orange)

Each half is cropped, scaled to fit 512x512 with its aspect ratio preserved
(bicubic), and centred on a black canvas -- the same 512x512 letterbox format as
every other image in ``ImageNet-single``. The crop boxes below were recovered by
fitting candidate boxes/filters against the committed files; both reproduce them
to within JPEG round-trip noise (MSE ~10, i.e. no visible difference).

Usage
-----
    # write the two halves next to the curated bookshelf folder
    python3 split_derived.py --out ImageNet-single-rebuilt

    # check the committed files still match the recipe
    python3 split_derived.py --verify
"""
import argparse
import os
import sys

from PIL import Image

DEFAULT_ORIGIN_DIR = "ImageNet"
DEFAULT_DATASET_DIR = "ImageNet"
DEFAULT_OUT_DIR = "ImageNet-rebuilt"

CANVAS = 512
BACKGROUND = (0, 0, 0)
RESAMPLE = Image.BICUBIC
# Matches how the rest of the dataset was encoded (standard q=75 tables, 4:2:0).
JPEG_KWARGS = {"quality": 75, "subsampling": 2}

# Recipe for every derived file: which origin, which crop box, which category.
# box is (left, upper, right, lower) in origin pixel coordinates.
RECIPES = [
    {
        "file": "n04190052_340_1.JPEG",
        "origin": "n04190052_340.JPEG",
        "category": "bookshelf",
        "box": (0, 0, 249, 332),
        "note": "left half - unpainted shelf (2px white divider at x=249..250 excluded)",
    },
    {
        "file": "n04190052_340_2.JPEG",
        "origin": "n04190052_340.JPEG",
        "category": "bookshelf",
        "box": (250, 0, 500, 331),
        "note": "right half - painted shelf",
    },
]


def letterbox(image, size=CANVAS):
    """Scale to fit a size x size square, keeping aspect, centred on black."""
    width, height = image.size
    scale = size / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = image.resize((new_width, new_height), RESAMPLE)
    canvas = Image.new("RGB", (size, size), BACKGROUND)
    canvas.paste(resized, ((size - new_width) // 2, (size - new_height) // 2))
    return canvas


def render(recipe, origin_dir):
    """Produce the 512x512 image described by a recipe."""
    origin_path = os.path.join(origin_dir, recipe["origin"])
    if not os.path.isfile(origin_path):
        raise FileNotFoundError(origin_path)
    with Image.open(origin_path) as origin:
        origin = origin.convert("RGB")
        box = recipe["box"]
        if box[2] > origin.width or box[3] > origin.height:
            raise ValueError(
                f"{recipe['origin']} is {origin.size}, too small for crop box {box}"
            )
        return letterbox(origin.crop(box))


def mean_squared_error(image_a, image_b):
    if image_a.size != image_b.size:
        return None
    pixels_a = image_a.tobytes()
    pixels_b = image_b.tobytes()
    total = sum((a - b) ** 2 for a, b in zip(pixels_a, pixels_b))
    return total / len(pixels_a)


def split(origin_dir, out_dir, flat):
    for recipe in RECIPES:
        image = render(recipe, origin_dir)
        dest_dir = out_dir if flat else os.path.join(out_dir, recipe["category"])
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, recipe["file"])
        image.save(dest, **JPEG_KWARGS)
        print(f"wrote {dest}  <- {recipe['origin']}{recipe['box']}  ({recipe['note']})")


def verify(origin_dir, dataset_dir):
    """Compare regenerated halves against the committed dataset files."""
    failures = 0
    for recipe in RECIPES:
        reference_path = os.path.join(dataset_dir, recipe["category"], recipe["file"])
        if not os.path.isfile(reference_path):
            print(f"[warn] no committed file to compare against: {reference_path}")
            continue
        regenerated = render(recipe, origin_dir)
        with Image.open(reference_path) as reference:
            error = mean_squared_error(regenerated, reference.convert("RGB"))
        if error is None:
            print(f"[FAIL] {recipe['file']}: size mismatch")
            failures += 1
        # JPEG re-encoding of an already-lossy source leaves a small residual;
        # anything under ~25 MSE is visually identical.
        elif error < 25:
            print(f"[ok]   {recipe['file']}: MSE={error:.2f}")
        else:
            print(f"[FAIL] {recipe['file']}: MSE={error:.2f} (recipe no longer matches)")
            failures += 1
    return failures


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--origin-dir",
        default=DEFAULT_ORIGIN_DIR,
        help="Folder holding the uncropped origin images (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help="Where to write the derived files (default: %(default)s).",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write straight into --out instead of --out/<category>/.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare against the committed dataset instead of writing files.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Curated dataset used by --verify (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verify:
        return 1 if verify(args.origin_dir, args.dataset_dir) else 0
    split(args.origin_dir, args.out, args.flat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
