"""Rename a folder of resized images back to their original curated names.

Reads the {curated name: resized name} dictionary produced by
match_resized_images.py, inverts it, and copies every image in the source
folder into a new folder under the curated name it came from.

Source filenames are not trusted on their own: this dataset contains at least
one file whose name collides with a different picture in the resized folder.
Each source image is therefore identified by its content, and its filename is
only used as a fallback when the resized folder is unavailable.

Usage:
    python rename_from_mapping.py
    python rename_from_mapping.py --source DIR --output DIR --move
"""

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image

from match_resized_images import MIN_CORRELATION, list_images, signature

MAPPING_FILE = "/media/isume/SSD-1/truck-dataset/curated_to_resized.json"
SOURCE_DIR = "/home/isume/test-env/Prompt2Craft/CraftGPT/dataset/ImageNet-single/truck"
OUTPUT_DIR = "/home/isume/test-env/Prompt2Craft/CraftGPT/dataset/ImageNet-single/truck_renamed"
RESIZED_DIR = "/media/isume/SSD-1/truck-dataset/resized"


def invert_mapping(mapping):
    """resized name -> curated name, flagging any resized name used twice."""
    reverse = {}
    duplicates = {}
    for curated, resized in mapping.items():
        if resized in reverse:
            duplicates.setdefault(resized, [reverse[resized]]).append(curated)
        else:
            reverse[resized] = curated
    return reverse, duplicates


def build_resized_index(resized_dir):
    """Signatures for the resized folder, so source files can be identified by content."""
    names = list_images(resized_dir)
    sigs = np.stack([
        signature(Image.open(os.path.join(resized_dir, n))) for n in names
    ])
    return names, sigs


def resolve_by_content(path, names, sigs, min_corr):
    """The resized filename whose picture this file actually holds, or None."""
    with Image.open(path) as im:
        scores = sigs @ signature(im)
    best = int(np.argmax(scores))
    if scores[best] < min_corr:
        return None, float(scores[best])
    return names[best], float(scores[best])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", default=MAPPING_FILE,
                        help="JSON mapping written by match_resized_images.py")
    parser.add_argument("--source", default=SOURCE_DIR, help="folder of files to rename")
    parser.add_argument("--output", default=OUTPUT_DIR, help="folder to write renamed files to")
    parser.add_argument("--move", action="store_true",
                        help="move instead of copy (source folder is emptied)")
    parser.add_argument("--use-source-ext", action="store_true",
                        help="keep the source file extension instead of the dictionary key's")
    parser.add_argument("--resized", default=RESIZED_DIR,
                        help="resized folder, used to identify source images by content")
    parser.add_argument("--no-verify", action="store_true",
                        help="trust source filenames instead of checking image content")
    parser.add_argument("--min-corr", type=float, default=MIN_CORRELATION,
                        help="minimum correlation to identify a source image "
                             "(default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen without writing anything")
    args = parser.parse_args()

    with open(args.dictionary) as f:
        mapping = json.load(f)
    reverse, duplicates = invert_mapping(mapping)
    print(f"{len(mapping)} dictionary entries, {len(reverse)} unique resized names")
    if duplicates:
        print(f"WARNING: {len(duplicates)} resized names map to several curated names; "
              "using the first of each:")
        for resized, curated in sorted(duplicates.items()):
            print(f"  {resized} <- {', '.join(curated)}")

    source_files = list_images(args.source)
    print(f"{len(source_files)} files in source folder")

    verify = not args.no_verify
    if verify:
        if not os.path.isdir(args.resized):
            parser.error(f"resized folder not found: {args.resized} "
                         "(pass --resized, or --no-verify to match on filenames alone)")
        print("Indexing resized folder to identify source images by content...")
        resized_names, resized_sigs = build_resized_index(args.resized)

    if not args.dry_run:
        os.makedirs(args.output, exist_ok=True)

    renamed = {}
    missing = []
    corrected = []
    for name in source_files:
        lookup = name
        if verify:
            actual, corr = resolve_by_content(
                os.path.join(args.source, name), resized_names, resized_sigs, args.min_corr)
            if actual is None:
                missing.append(name)
                continue
            if actual != name:
                corrected.append((name, actual, corr))
            lookup = actual

        curated = reverse.get(lookup)
        if curated is None:
            missing.append(name)
            continue

        if args.use_source_ext:
            new_name = os.path.splitext(curated)[0] + os.path.splitext(name)[1]
        else:
            new_name = curated

        renamed[name] = new_name
        if args.dry_run:
            continue

        src = os.path.join(args.source, name)
        dst = os.path.join(args.output, new_name)
        if args.move:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)

    verb = "Would rename" if args.dry_run else ("Moved" if args.move else "Copied")
    print(f"\n{verb} {len(renamed)}/{len(source_files)} files -> {args.output}")
    for name in list(renamed)[:5]:
        print(f"  {name} -> {renamed[name]}")
    if len(renamed) > 5:
        print(f"  ... and {len(renamed) - 5} more")

    if corrected:
        print(f"\nSource files whose name disagreed with their content ({len(corrected)}); "
              "renamed from what the picture actually is:")
        for name, actual, corr in corrected:
            print(f"  {name} holds {actual} (corr {corr:.5f}) -> {renamed.get(name, '<no entry>')}")

    if missing:
        print(f"\nNot in the dictionary ({len(missing)}), left untouched:")
        print("  " + ", ".join(missing))


if __name__ == "__main__":
    main()
