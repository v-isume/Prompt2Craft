"""Screen generated crafts for parts that intersect, or that fail to touch.

Runs between mesh generation and ``check_physics.py``, over the meshes on disk:

* **collisions** -- any two parts whose bounding boxes interpenetrate by more
  than the depth threshold, so parts that merely touch or graze each other pass.
  Parts joined by a boolean operation are meant to overlap and are excluded.
  Written to ``failures.json``.
* **floating parts** -- any two parts a plan says are connected whose bounding
  boxes do not reach each other. Written to ``float.json``.

Both files are what ``check_physics.py`` skips on and what the
``reprompt_collision.py`` feedback quotes, so this is the stage that decides
which crafts are worth simulating. A craft already flagged as colliding is not
checked for floating parts.

Runtime settings default to ``config/check_collisions_config.json`` and may be
overridden on the command line.

Usage:
    python check_collisions.py
    python check_collisions.py --obj-type chair --in-dir new/initial
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import trimesh

from config_loader import load_config
from craft_assembly import aabb_separated, lookup_record


class CollisionManager:
    """Pairwise bounding-box overlap between the parts of one craft.

    Bounding boxes rather than exact meshes: the parts are axis-aligned boxes
    and cylinders, and this runs over every pair of every craft, so the cheap
    test is both accurate enough and fast enough.
    """

    def __init__(self):
        self.names = []
        self.meshes = []

    def add_object(self, name, mesh):
        self.names.append(name)
        self.meshes.append(mesh)

    def remove_object(self, name):
        if name in self.names:
            index = self.names.index(name)
            self.names.pop(index)
            self.meshes.pop(index)

    def collisions(self):
        """Overlapping pairs with their depth, deepest first."""
        found = []
        for i in range(len(self.meshes)):
            for j in range(i + 1, len(self.meshes)):
                bounds_i = self.meshes[i].bounds
                bounds_j = self.meshes[j].bounds
                if all(bounds_i[1] > bounds_j[0]) and all(bounds_i[0] < bounds_j[1]):
                    found.append((
                        [self.names[i], self.names[j]],
                        penetration_depth(self.meshes[i], self.meshes[j]),
                    ))
        return sorted(found, key=lambda entry: entry[1], reverse=True)


def penetration_depth(mesh_a, mesh_b):
    """How far two overlapping parts would have to move to come apart, in mm.

    The boxes intersect along each axis over ``[max(lower), min(upper)]``, and
    the shortest of those three spans is the smallest move that pulls them
    apart -- the figure a tolerance in millimetres should be compared against.

    Intersecting the boxes directly, rather than comparing centre distances,
    is what keeps this right when one part sits entirely inside another: a thin
    plate buried in a block penetrates by its own thickness, not by the block's.
    It also reads off ``bounds`` rather than ``mesh.centroid``, which for a part
    with a hole cut through it no longer sits in the middle of its box.
    """
    lower = np.maximum(mesh_a.bounds[0], mesh_b.bounds[0])
    upper = np.minimum(mesh_a.bounds[1], mesh_b.bounds[1])
    return float(np.min(upper - lower))


def load_parts(craft_dir):
    """Every part mesh of one craft, keyed by the part name."""
    manager = CollisionManager()
    for part_file in os.listdir(craft_dir):
        mesh = trimesh.load(os.path.join(craft_dir, part_file))
        manager.add_object(part_file.split(".")[0], mesh)
    return manager


def colliding_pairs(manager, boolean_pairs, threshold):
    """Overlaps that count as failures: deep enough, and not a boolean pair.

    Parts are allowed to touch, and a part inserted into another is *meant* to
    intersect it, so both are filtered out.
    """
    failures = []
    for pair, depth in manager.collisions():
        if depth <= threshold:
            continue
        ordered = sorted(pair)
        if any(ordered[0] in bool_op and ordered[1] in bool_op for bool_op in boolean_pairs):
            continue
        failures.append(pair)
    return failures


def floating_pairs(craft_dir, connection_pairs, tol):
    """Connected pairs whose meshes never reach each other."""
    floating = []
    for pair in connection_pairs:
        meshes = [trimesh.load(os.path.join(craft_dir, name + ".obj")) for name in pair[:2]]
        if aabb_separated(meshes[0].bounds, meshes[1].bounds, tol):
            floating.append(pair)
    return floating


def write_json(path, payload, label):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as file:
        file.write(json.dumps(payload))
    print(f"{len(payload)} crafts with {label} -> {path}")


def parse_args(defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj-type", default=defaults["obj_type"])
    parser.add_argument("--results-root", default=defaults["results_root"])
    parser.add_argument("--in-dir", default=defaults["in_dir"],
                        help="pipeline round holding the meshes and pose records")
    parser.add_argument("--records-suffix", default=defaults["records_suffix"],
                        help="suffix on the folder the boolean/connection tables live in")
    parser.add_argument("--depth-threshold", type=float,
                        default=defaults["collision_depth_threshold"],
                        help="overlap below this counts as touching, not colliding")
    parser.add_argument("--float-tolerance", type=float, default=defaults["float_tolerance"])
    parser.add_argument("--ignore-id", action="append",
                        default=list(defaults.get("ignore_ids", [])))
    return parser.parse_args()


def main():
    defaults = load_config("check_collisions_config.json")
    args = parse_args(defaults)

    root = f"{args.results_root}/{args.in_dir}"
    meshes_dir = f"{root}/meshes/{args.obj_type}"
    records_dir = f"{root}/error/{args.obj_type}{args.records_suffix}"
    output_dir = f"{root}/error/{args.obj_type}"
    ignore_ids = set(args.ignore_id)

    boolean_pd = pd.read_json(os.path.join(records_dir, "boolean.jsonl"), lines=True)
    connection_pd = pd.read_json(os.path.join(records_dir, "connections.jsonl"), lines=True)

    all_crafts = sorted(os.listdir(meshes_dir))
    collision_dict = {}
    float_dict = {}

    for custom_id in all_crafts:
        if custom_id in ignore_ids:
            continue
        print(custom_id)

        craft_dir = os.path.join(meshes_dir, custom_id)
        boolean_record = lookup_record(boolean_pd, custom_id, "pairs")
        # A craft with no boolean record simply has no inserted parts to exclude.
        boolean_pairs = boolean_record[0] if boolean_record else []

        failures = colliding_pairs(load_parts(craft_dir), boolean_pairs, args.depth_threshold)
        if failures:
            collision_dict[custom_id] = failures
            # A craft that already fails on collisions is not worth measuring
            # for floating parts; the collision feedback comes first.
            continue

        connection_record = lookup_record(connection_pd, custom_id, "connection_pairs")
        if connection_record is None:
            print(f"  no connection record for {custom_id}, skipping the floating check")
            continue

        floating = floating_pairs(craft_dir, connection_record[0], args.float_tolerance)
        if floating:
            float_dict[custom_id] = floating

    print()
    write_json(os.path.join(output_dir, "failures.json"), collision_dict, "colliding parts")
    write_json(os.path.join(output_dir, "float.json"), float_dict, "floating parts")


if __name__ == "__main__":
    main()
