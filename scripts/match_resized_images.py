"""Match curated ImageNet images to their already-resized 512x512 counterparts.

Each curated image is resized to 512x512 the same way the target folder was built
(aspect ratio preserved, letterboxed with black padding), then compared against
every image in the resized folder. Matches are written out as a dictionary
{curated filename: resized filename}.

Usage:
    python match_resized_images.py
    python match_resized_images.py --curated DIR --resized DIR --output FILE
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageOps

CURATED_DIR = "/media/isume/SSD-1/truck-dataset/ImageNet/truck/curated"
RESIZED_DIR = "/media/isume/SSD-1/truck-dataset/resized"
OUTPUT_FILE = "/media/isume/SSD-1/truck-dataset/curated_to_resized.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SIZE = (512, 512)
SIG_SIZE = 16  # signature is a SIG_SIZE x SIG_SIZE grayscale thumbnail
TOP_K = 5  # candidates per image to verify at full resolution

# Signature correlation decides whether two images are the same picture. On this
# dataset genuine pairs land at >= 0.9989 and the nearest non-pair at 0.92, so
# anything in between is a safe cutoff. Plain pixel difference is a poor
# criterion here: it scales with how detailed the image is, not with similarity.
MIN_CORRELATION = 0.99


def list_images(folder):
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        and os.path.isfile(os.path.join(folder, f))
    )


def resize_to_square(image, size=SIZE):
    """Resize preserving aspect ratio, padding the remainder with black."""
    return ImageOps.pad(image, size, method=Image.LANCZOS, color=(0, 0, 0))


def signature(image):
    """Mean-removed, unit-norm grayscale thumbnail; dot product = correlation."""
    thumb = np.asarray(
        image.convert("L").resize((SIG_SIZE, SIG_SIZE), Image.LANCZOS),
        dtype=np.float32,
    )
    thumb -= thumb.mean()
    norm = np.linalg.norm(thumb)
    return thumb.ravel() / (norm if norm else 1.0)


def load_pixels(path):
    """Full-resolution grayscale pixels, for verifying a candidate match."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize(SIZE, Image.LANCZOS), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated", default=CURATED_DIR, help="folder of source images")
    parser.add_argument("--resized", default=RESIZED_DIR, help="folder of 512x512 images")
    parser.add_argument("--output", default=OUTPUT_FILE, help="JSON file to write")
    parser.add_argument("--save-resized", default=None,
                        help="optional folder to also write the resized curated images to")
    parser.add_argument("--min-corr", type=float, default=MIN_CORRELATION,
                        help="minimum signature correlation for a match (default: %(default)s)")
    args = parser.parse_args()

    curated_files = list_images(args.curated)
    resized_files = list_images(args.resized)
    print(f"{len(curated_files)} curated images, {len(resized_files)} resized images")

    if args.save_resized:
        os.makedirs(args.save_resized, exist_ok=True)

    # Index the resized folder once.
    print("Indexing resized folder...")
    resized_sigs = np.stack([
        signature(Image.open(os.path.join(args.resized, f))) for f in resized_files
    ])
    pixel_cache = {}

    matches = {}
    unmatched = []

    for i, name in enumerate(curated_files, 1):
        with Image.open(os.path.join(args.curated, name)) as im:
            square = resize_to_square(im.convert("RGB"))

        if args.save_resized:
            square.save(os.path.join(args.save_resized, f"{os.path.splitext(name)[0]}.jpg"),
                        quality=95)

        scores = resized_sigs @ signature(square)
        candidates = [i for i in np.argsort(scores)[::-1][:TOP_K]
                      if scores[i] >= args.min_corr]

        # Several candidates can clear the bar when the dataset holds near
        # duplicates; the full-resolution pixel difference picks the right one.
        best_name, best_mae = None, float("inf")
        square_pixels = np.asarray(square.convert("L"), dtype=np.float32)
        for idx in candidates:
            other = resized_files[idx]
            if other not in pixel_cache:
                pixel_cache[other] = load_pixels(os.path.join(args.resized, other))
            mae = float(np.abs(square_pixels - pixel_cache[other]).mean())
            if mae < best_mae:
                best_name, best_mae = other, mae

        if best_name is not None:
            matches[name] = best_name
        else:
            unmatched.append((name, float(scores.max())))

        if i % 100 == 0 or i == len(curated_files):
            print(f"  {i}/{len(curated_files)} processed, {len(matches)} matched")

    with open(args.output, "w") as f:
        json.dump(matches, f, indent=2, sort_keys=True)

    print(f"\nMatched {len(matches)}/{len(curated_files)} images -> {args.output}")

    if unmatched:
        print(f"\nNo match ({len(unmatched)}), with their best correlation:")
        for name, corr in sorted(unmatched, key=lambda x: -x[1]):
            print(f"  {corr:.4f}  {name}")

    # Two curated images mapping to one resized file means a real duplicate or a
    # bad match, so surface it rather than letting the dict silently lose one.
    targets = {}
    for src, dst in matches.items():
        targets.setdefault(dst, []).append(src)
    collisions = {d: s for d, s in targets.items() if len(s) > 1}
    if collisions:
        print(f"\nResized files claimed by more than one curated image ({len(collisions)}):")
        for dst, srcs in sorted(collisions.items()):
            print(f"  {dst} <- {', '.join(srcs)}")

    leftover = [f for f in resized_files if f not in targets]
    if leftover:
        print(f"\nResized images with no curated source ({len(leftover)}): "
              f"{', '.join(leftover[:10])}" + (" ..." if len(leftover) > 10 else ""))


if __name__ == "__main__":
    main()
