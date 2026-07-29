#!/usr/bin/env python3
"""Reproducible construction of the ImageNet-single dataset.

The curated ``ImageNet-single/<category>`` folders were built by hand-picking
images out of downloaded ImageNet synsets (plus a few external stock photos).
This tool makes that selection reproducible in two steps:

  1. ``snapshot`` scans the existing curated folders and writes a manifest
     (``dataset_manifest.json``) that records, per category, exactly which
     files were kept and where each one came from. Commit this manifest.

  2. ``rebuild`` takes a folder of downloaded ImageNet synsets plus the
     manifest and recreates the category folders by copying the listed files.

Each kept file is classified as:

  * ``imagenet``  – named ``n<synset>_<id>.<ext>`` (e.g. ``n02924116_1127.JPEG``).
                    Directly reproducible: copied verbatim from the synset pool.
  * ``derived``   – named ``n<synset>_<id>_<k>.<ext>`` (e.g. ``n04190052_340_1.JPEG``).
                    A manual crop/variant of the synset origin ``n<synset>_<id>.<ext>``.
                    Its exact pixels can't be regenerated automatically; the manifest
                    records the origin so the manual step is documented.
  * ``external``  – anything else (Pexels stock photos, sequentially renamed files).
                    Not part of ImageNet; cannot be sourced from the synset pool.

Only ``imagenet`` files are copied by ``rebuild``; ``derived`` and ``external``
files are reported so it is explicit what still needs its original source.
"""
import argparse
import json
import os
import re
import shutil
from collections import OrderedDict

DEFAULT_CATEGORIES = [
    "bookshelf", "bus", "chair", "hammer",
    "scooter", "skateboard", "table", "truck",
]
DEFAULT_DATASET_DIR = "./dataset/ImageNet/synsets"
DEFAULT_MANIFEST = "./config/dataset_manifest.json"

# n<synset>_<id>.<ext>            -> directly from a downloaded synset
IMAGENET_RE = re.compile(r"^(n\d+)_\d+\.[A-Za-z]+$")
# n<synset>_<id>_<k>.<ext>        -> hand-derived variant of a synset image
DERIVED_RE = re.compile(r"^(n\d+_\d+)_\d+\.([A-Za-z]+)$")


def classify(filename):
    """Return (kind, synset, origin) for a filename.

    kind is one of "imagenet", "derived", "external". synset/origin are None
    when not applicable.
    """
    m = IMAGENET_RE.match(filename)
    if m:
        return "imagenet", m.group(1), None
    m = DERIVED_RE.match(filename)
    if m:
        base, ext = m.group(1), m.group(2)
        synset = base.split("_", 1)[0]
        return "derived", synset, f"{base}.{ext}"
    return "external", None, None


def list_images(folder):
    return sorted(
        name for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
    )


def snapshot(dataset_dir, categories, manifest_path):
    manifest = OrderedDict()
    manifest["generated_from"] = dataset_dir
    manifest["categories"] = OrderedDict()

    for category in categories:
        folder = os.path.join(dataset_dir, category)
        if not os.path.isdir(folder):
            print(f"[warn] category folder not found, skipping: {folder}")
            continue

        imagenet_files, derived_files, external_files = [], [], []
        synsets = set()
        for name in list_images(folder):
            kind, synset, origin = classify(name)
            if kind == "imagenet":
                imagenet_files.append(name)
                synsets.add(synset)
            elif kind == "derived":
                derived_files.append({"file": name, "origin": origin})
                synsets.add(synset)
            else:
                external_files.append(name)

        entry = OrderedDict()
        entry["synsets"] = sorted(synsets)
        entry["counts"] = {
            "imagenet": len(imagenet_files),
            "derived": len(derived_files),
            "external": len(external_files),
            "total": len(imagenet_files) + len(derived_files) + len(external_files),
        }
        entry["imagenet"] = imagenet_files
        entry["derived"] = derived_files
        entry["external"] = external_files
        manifest["categories"][category] = entry
        print(
            f"{category:<12} total={entry['counts']['total']:>3}  "
            f"imagenet={entry['counts']['imagenet']:>3}  "
            f"derived={entry['counts']['derived']:>2}  "
            f"external={entry['counts']['external']:>3}  "
            f"synsets={entry['synsets']}"
        )

    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2)
    print(f"\nWrote manifest: {manifest_path}")


def index_source(source_dir):
    """Map filename -> full path for every file under source_dir (recursive)."""
    index = {}
    duplicates = 0
    for root, _dirs, files in os.walk(source_dir):
        for name in files:
            if name in index:
                duplicates += 1
                continue
            index[name] = os.path.join(root, name)
    if duplicates:
        print(f"[warn] {duplicates} duplicate filename(s) in source; kept first occurrence")
    return index


def rebuild(source_dir, out_dir, manifest_path, categories=None):
    with open(manifest_path) as file:
        manifest = json.load(file)

    print(f"Indexing source synsets: {source_dir}")
    index = index_source(source_dir)
    print(f"Indexed {len(index)} source files\n")

    report = OrderedDict()
    all_categories = manifest["categories"]
    wanted = categories or list(all_categories.keys())

    for category in wanted:
        entry = all_categories.get(category)
        if entry is None:
            print(f"[warn] category not in manifest, skipping: {category}")
            continue

        dest = os.path.join(out_dir, category)
        os.makedirs(dest, exist_ok=True)

        copied, missing = [], []
        for name in entry["imagenet"]:
            src = index.get(name)
            if src is None:
                missing.append(name)
                continue
            shutil.copy2(src, os.path.join(dest, name))
            copied.append(name)

        derived = entry.get("derived", [])
        external = entry.get("external", [])
        report[category] = {
            "copied": copied,
            "missing_from_source": missing,
            "derived_not_reproduced": derived,
            "external_not_reproduced": external,
        }

        # Write a per-category note listing everything that could not be reproduced.
        unresolved = missing or derived or external
        if unresolved:
            with open(os.path.join(dest, "_UNRESOLVED.txt"), "w") as note:
                if missing:
                    note.write("# ImageNet files listed in manifest but absent from source:\n")
                    note.writelines(f"{n}\n" for n in missing)
                if derived:
                    note.write("\n# Hand-derived crops (need origin + manual crop step):\n")
                    note.writelines(f"{d['file']}  <- {d['origin']}\n" for d in derived)
                if external:
                    note.write("\n# External (non-ImageNet) images, original source required:\n")
                    note.writelines(f"{n}\n" for n in external)

        print(
            f"{category:<12} copied={len(copied):>3}/{len(entry['imagenet']):<3}  "
            f"missing={len(missing):>3}  derived={len(derived):>2}  external={len(external):>3}"
        )

    report_path = os.path.join(out_dir, "rebuild_report.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w") as file:
        json.dump(report, file, indent=2)
    print(f"\nRebuilt dataset under: {out_dir}")
    print(f"Wrote report: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_snap = sub.add_parser("snapshot", help="Record the current curated folders into a manifest.")
    p_snap.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p_snap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    p_snap.add_argument("--manifest", default=DEFAULT_MANIFEST)

    p_build = sub.add_parser("rebuild", help="Recreate the folders from a synset pool + manifest.")
    p_build.add_argument("--source", required=True, help="Folder containing the downloaded ImageNet synsets.")
    p_build.add_argument("--out", default="./dataset/ImageNet/rebuilt")
    p_build.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p_build.add_argument("--categories", nargs="+", default=None, help="Subset to rebuild (default: all in manifest).")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "snapshot":
        snapshot(args.dataset_dir, args.categories, args.manifest)
    elif args.command == "rebuild":
        rebuild(args.source, args.out, args.manifest, args.categories)


if __name__ == "__main__":
    main()
