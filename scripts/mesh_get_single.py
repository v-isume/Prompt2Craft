"""Build the meshes for a single assembly plan.

Same job as ``batch_mesh_gen_streamlined.py``, but for one plan at a time and
without the error bookkeeping -- the plan is walked into a connection tree, the
parts are placed by following it, and each part is exported as its own ``.obj``.

This reads the ``_MOD`` plan dialect produced by ``prompt_single_robot.py``:
modifications name the face they are cut from (``FRONT_MOD`` ...) and insertions
carry an explicit ``insertion_depth``. That is a different vocabulary from the
one ``craft_assembly.py`` implements, which is why the geometry here is
deliberately kept separate rather than shared with it.

Runtime settings default to ``config/mesh_single_config.json`` and may be
overridden on the command line.

Usage:
    python mesh_get_single.py
    python mesh_get_single.py --obj-name 0003 --output-dir DIR
"""
import argparse
import json
import os

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from config_loader import load_available_dims, load_config

# Index into CraftPart.occupied, which tracks how many parts already sit against
# each face of a part at a given alignment.
FACE_INDEX = {"FRONT": 0, "BACK": 1, "RIGHT": 2, "LEFT": 3, "TOP": 4, "BOTTOM": 5}

FACE_OPPOSITE = {
    "FRONT": "BACK",
    "BACK": "FRONT",
    "RIGHT": "LEFT",
    "LEFT": "RIGHT",
    "TOP": "BOTTOM",
    "BOTTOM": "TOP",
}

# The face a contact is on decides which axis the parts are stacked along, and
# in which direction: (axis, sign).
CONTACT_FACE_AXIS = {
    "FRONT": (0, +1),
    "BACK": (0, -1),
    "RIGHT": (1, +1),
    "LEFT": (1, -1),
    "TOP": (2, +1),
    "BOTTOM": (2, -1),
}

# Plain alignment keywords per axis -> (sign of the flush offset, occupancy slot).
# CENTER and CONTACT leave the part centred on that axis.
PLAIN_ALIGN = {
    0: {"FRONT": (+1, 0), "BACK": (-1, 1), "CENTER": (0, 2), "CONTACT": (0, 2)},
    1: {"RIGHT": (+1, 0), "LEFT": (-1, 1), "CENTER": (0, 2), "CONTACT": (0, 2)},
    2: {"HIGH": (+1, 0), "LOW": (-1, 1), "CENTER": (0, 2), "CONTACT": (0, 2)},
}

# Modification keywords -> (axis, sign, face drilled, face opposite it).
MOD_ALIGN = {
    "FRONT_MOD": (0, +1, "FRONT", "BACK"),
    "BACK_MOD": (0, -1, "BACK", "FRONT"),
    "RIGHT_MOD": (1, +1, "RIGHT", "LEFT"),
    "LEFT_MOD": (1, -1, "LEFT", "RIGHT"),
    "TOP_MOD": (2, +1, "TOP", "BOTTOM"),
    "BOTTOM_MOD": (2, -1, "BOTTOM", "TOP"),
}

# Mirrored keywords, for when an insertion has to be resolved from the other
# side because the part it names has already been placed.
MIRROR_ALIGN = {
    "FRONT": "BACK", "BACK": "FRONT",
    "RIGHT": "LEFT", "LEFT": "RIGHT",
    "HIGH": "LOW", "LOW": "HIGH",
    "FRONT_MOD": "BACK_MOD", "BACK_MOD": "FRONT_MOD",
    "RIGHT_MOD": "LEFT_MOD", "LEFT_MOD": "RIGHT_MOD",
    "TOP_MOD": "BOTTOM_MOD", "BOTTOM_MOD": "TOP_MOD",
}

# How deep an inserted part goes, as a fraction of the reference length.
# "ACROSS" is handled separately: it centres the part on the modification.
INSERTION_DEPTH = {"ACROSS": 1.00, "QUARTER": 0.25, "HALF": 0.50, "FULL": 1.00}

# Euler angles for each orthogonal re-ordering of a cuboid's dimensions, keyed so
# that chosen_obj[i] == orientation[perm[i]]. Note this is the inverse of the
# convention craft_assembly.CUBOID_EULER_BY_PERM uses.
CUBOID_EULER_BY_PERM = {
    (0, 1, 2): [0, 0, 0],
    (1, 0, 2): [0, 0, 90],
    (2, 1, 0): [0, 90, 0],
    (0, 2, 1): [90, 0, 0],
    (1, 2, 0): [90, 0, 90],
    (2, 0, 1): [0, 90, 90],
}

# Cylinder orientation -> (euler angles, order the dimensions end up in).
CYLINDER_ROTATION = {
    "LEFT_RIGHT": ([90, 0, 0], (0, 2, 1)),
    "FRONT_BACK": ([0, 90, 0], (2, 0, 1)),
    "TOP_BOTTOM": ([0, 0, 0], (0, 1, 2)),
}

available_dims = load_available_dims()


class CraftPart:
    def __init__(self, part_name, chosen_obj, orientation):
        # Initialize part with the name, selected obj and orientation
        self.name = part_name
        self.chosen_obj = chosen_obj
        self.orientation = orientation
        # Variables to keep track of state of the object
        self.translated = False
        self.current_dims = []
        self.rotation = [0, 0, 0]
        self.position = np.array([0.0, 0.0, 0.0])
        self.shape = chosen_obj[3]
        # FRONT, BACK, RIGHT, LEFT, TOP, BOTTOM - align_x, align_y, align_z
        self.occupied = np.zeros((6, 3, 3, 3))

    def initialize_part(self):
        """Resolve the requested orientation into a rotation and final dimensions."""
        if self.shape == "cuboid":
            for perm, rot_euler in CUBOID_EULER_BY_PERM.items():
                if all(self.chosen_obj[i] == self.orientation[perm[i]] for i in range(3)):
                    self.current_dims = self.orientation
                    self.rotation = rot_euler
                    return
            raise ValueError(
                f"The ORIENTATION of {self.name} ({self.orientation}) is not an orthogonal "
                f"re-ordering of {self.chosen_obj[:3]}."
            )

        if self.shape == "cylinder":
            if self.orientation not in CYLINDER_ROTATION:
                raise ValueError(
                    f"The ORIENTATION of {self.name} ({self.orientation}) is not valid. "
                    f"Should be one of {sorted(CYLINDER_ROTATION)}."
                )
            rot_euler, axis_order = CYLINDER_ROTATION[self.orientation]
            self.current_dims = [self.chosen_obj[i] for i in axis_order]
            self.rotation = rot_euler
            return

        raise ValueError(f"Unknown shape type: {self.shape}")

    def translate(self, translation):
        self.position += translation
        self.translated = True

    def build_mesh(self):
        """The part's mesh, rotated and moved to its final pose."""
        if self.shape == "cuboid":
            mesh = trimesh.creation.box(self.current_dims)
            mesh.apply_translation(-mesh.centroid)
        elif self.shape == "cylinder":
            mesh = trimesh.creation.cylinder(
                radius=self.chosen_obj[0] / 2, height=self.chosen_obj[2])
            mesh.apply_translation(-mesh.centroid)
            transform = np.eye(4)
            transform[:3, :3] = R.from_euler("xyz", self.rotation, degrees=True).as_matrix()
            mesh.apply_transform(transform)
        else:
            raise ValueError(f"Unknown shape type: {self.shape}")

        mesh.apply_translation(self.position)
        return mesh


class CraftAssembly:
    def __init__(self, img_name):
        self.img_name = img_name
        self.connections = []
        self.modifications = []
        self.list_parts = []
        self.boolean_pairs = []

    def add_part(self, craft_part):
        self.list_parts.append(craft_part)

    def add_connection(self, connection_pair):
        """Record a connection, unless the pair is already connected.

        An existing INSERTED connection is replaced when the modification it
        names belongs to the other part of the pair: that phrasing is the one
        the geometry can act on.
        """
        replace = False
        index = None
        for pair in self.connections:
            if connection_pair[0] in pair and connection_pair[1] in pair:
                if pair[2]["contact_type"] == "INSERTED":
                    mod_name = pair[2]["to_modification"]
                    for mod in self.modifications:
                        if connection_pair[1] == mod[0] and mod_name == mod[1]["name"]:
                            replace = True
                            index = self.connections.index(pair)
                            break
                    if replace:
                        break
                    return

        if replace:
            self.connections.pop(index)
        self.connections.append(connection_pair)

    def add_modification(self, modification_pair):
        self.modifications.append(modification_pair)

    def get_by_name(self, name):
        for part in self.list_parts:
            if name == part.name:
                return part
        return None

    def update_tree(self, node, tree, ignore_idx=None, ordered_parts=None):
        if ignore_idx is None:
            ignore_idx = []
        if ordered_parts is None:
            ordered_parts = []
        for i in range(len(self.connections)):
            if node in self.connections[i] and i not in ignore_idx:
                tree.append(self.connections[i])
                ordered_parts.append(node)
                ignore_idx.append(i)
                if node == self.connections[i][0]:
                    self.update_tree(self.connections[i][1], tree, ignore_idx, ordered_parts)
                else:
                    self.update_tree(self.connections[i][0], tree, ignore_idx, ordered_parts)

    def generate_tree(self):
        """Walk the connections from the first part outwards, placing order first."""
        tree = []
        ordered_parts = []
        self.update_tree(self.list_parts[0].name, tree, ignore_idx=[], ordered_parts=ordered_parts)

        # validation
        for part in self.list_parts:
            if not any(part.name in conn for conn in tree):
                print(f"The assembly tree is not valid. {part.name} is not in the tree.")

        return tree, ordered_parts

    def connect_surface(self, main_part, move_part, connection):
        """Place ``move_part`` flush against one face of ``main_part``."""
        # The connection always names the face from the first part's point of
        # view, so it has to be flipped when the other part is the fixed one.
        contact_face = connection[2]["to_face"]
        if main_part.name != connection[0]:
            contact_face = FACE_OPPOSITE[contact_face]

        main_dims = main_part.current_dims
        move_dims = move_part.current_dims

        axis, sign = CONTACT_FACE_AXIS[contact_face]
        translation = np.array(main_part.position, dtype=float)
        translation[axis] += sign * (main_dims[axis] / 2 + move_dims[axis] / 2)

        slots = apply_alignments(translation, connection[2], main_dims, move_dims)
        occupy(main_part, move_part, [FACE_INDEX[contact_face]], slots, main_dims, move_dims)
        move_part.translate(translation)

    def connect_insert(self, connection):
        """Place one part inside a modification cut into another."""
        main_part = self.get_by_name(connection[1])
        move_part = self.get_by_name(connection[0])

        main_mod = None
        for mod in self.modifications:
            if main_part.name == mod[0] and mod[1]["name"] == connection[2]["to_modification"]:
                main_mod = mod
                break
        if main_mod is None:
            print("Error: No corresponding modification was found")
            return

        alignments = {axis: main_mod[1][axis] for axis in ("align_x", "align_y", "align_z")}

        # The part holding the modification may already be placed, in which case
        # the other one has to move instead and every alignment is mirrored.
        boolean_invert = move_part.translated
        if boolean_invert:
            main_part, move_part = move_part, main_part
            alignments = {a: MIRROR_ALIGN.get(v, v) for a, v in alignments.items()}

        if move_part.translated and main_part.translated:
            print(f"error, cyclical connection: {connection[0]} - {connection[1]}")
            return

        main_dims = main_part.current_dims
        move_dims = move_part.current_dims
        depth = INSERTION_DEPTH[connection[2]["insertion_depth"]]
        across = connection[2]["insertion_depth"] == "ACROSS"

        translation = np.array(main_part.position - move_part.position, dtype=float)
        slots, faces = apply_alignments(
            translation, alignments, main_dims, move_dims,
            insertion=(depth, across, boolean_invert),
        )

        occupy(main_part, move_part, faces, slots, main_dims, move_dims)
        move_part.translate(translation)

        if not boolean_invert:
            self.boolean_pairs.append((main_part, move_part))
        else:
            self.boolean_pairs.append((move_part, main_part))

    def save_parts(self, out_dir):
        for part in self.list_parts:
            part.build_mesh().export(os.path.join(out_dir, part.name + ".obj"))


def apply_alignments(translation, alignments, main_dims, move_dims, insertion=None):
    """Offset ``translation`` by the three alignments, in place.

    Returns the occupancy slot per axis, and -- for insertions -- the faces the
    modification takes up.
    """
    slots = []
    faces = []
    for axis, key in enumerate(("align_x", "align_y", "align_z")):
        align = alignments[key]

        if insertion is not None and align in MOD_ALIGN:
            mod_axis, sign, face, opposite = MOD_ALIGN[align]
            depth, across, boolean_invert = insertion
            faces.append(FACE_INDEX[face])
            if across:
                # Centred on the modification: it takes up both faces.
                faces.append(FACE_INDEX[opposite])
            else:
                # The part slides in until `depth` of the shorter of the two
                # lengths is buried -- or the longer one, seen from the far side.
                reference = (max(main_dims[mod_axis], move_dims[mod_axis]) if boolean_invert
                             else min(main_dims[mod_axis], move_dims[mod_axis]))
                translation[mod_axis] += sign * (
                    main_dims[mod_axis] / 2 + move_dims[mod_axis] / 2 - depth * reference
                )
            slots.append(2)
            continue

        if align not in PLAIN_ALIGN[axis]:
            raise ValueError(f"The {key} value ({align}) is not valid for this axis.")
        sign, slot = PLAIN_ALIGN[axis][align]
        translation[axis] += sign * (main_dims[axis] / 2 - move_dims[axis] / 2)
        slots.append(slot)

    return (slots, faces) if insertion is not None else slots


def occupy(main_part, move_part, faces, slots, main_dims, move_dims):
    """Claim the slots this connection uses, nudging aside if already taken.

    Two parts asking for the same face and alignment would land on top of each
    other, so the second one is pushed clear along whichever axes are not
    centred. A centred clash has no free direction, and is only reported.
    """
    if not faces:
        raise ValueError(
            f"The modification inserting into {main_part.name} does not specify a drilling "
            "direction: one of align_x, align_y or align_z must use a \"_MOD\" value."
        )

    idx_x, idx_y, idx_z = slots
    if main_part.occupied[faces[0]][idx_x][idx_y][idx_z] != 0:
        offset = np.zeros(3, dtype=float)
        for axis, slot in enumerate(slots):
            if slot == 2:
                print(f"error, same place in {'xyz'[axis]}", main_part.name)
                continue
            sign = +1 if slot == 0 else -1
            offset[axis] = sign * (move_dims[axis] / 2 - main_dims[axis] / 4)
        move_part.translate(offset)

    for face in faces:
        main_part.occupied[face][idx_x][idx_y][idx_z] += 1


def build_assembly(answer, img_name):
    """Turn a plan into an assembly with every part, modification and connection."""
    craft = CraftAssembly(img_name)

    for part_name, part_spec in answer.items():
        part = CraftPart(
            part_name=part_name,
            chosen_obj=available_dims[part_spec["AVAILABLE_OBJ"]],
            orientation=part_spec["ORIENTATION"],
        )
        part.initialize_part()
        craft.add_part(part)

        for mod in part_spec.get("modifications", []):
            craft.add_modification((part_name, mod))

        for conn in part_spec.get("connections", []):
            # A part cannot be inserted into a modification of itself; those
            # entries describe the hole, not a connection to resolve.
            if "to_modification" in conn:
                if conn["to_modification"] == "":
                    continue
                own_mod = any(mod[0] == part_name and conn["to_modification"] == mod[1]["name"]
                              for mod in craft.modifications)
                if own_mod:
                    continue
            craft.add_connection((part_name, conn["to_part"], conn))

    return craft


def place_parts(craft, tree, ordered_parts):
    """Follow the tree, moving each part into contact with the one before it."""
    for c_pair, main_p in zip(tree, ordered_parts):
        move_name = c_pair[1] if main_p == c_pair[0] else c_pair[0]
        move_part = craft.get_by_name(move_name)
        main_part = craft.get_by_name(main_p)

        for conn in craft.connections:
            if c_pair[0] in conn and c_pair[1] in conn:
                if conn[2]["contact_type"] == "SURFACE":
                    craft.connect_surface(main_part, move_part, c_pair)
                elif conn[2]["contact_type"] == "INSERTED":
                    craft.connect_insert(c_pair)

                print(f"{c_pair} was added.")
                print("----------------------------------")


def parse_args(defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj-name", default=defaults["obj_name"])
    parser.add_argument("--img-name", default=defaults["img_name"])
    parser.add_argument("--answer-json", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    defaults = load_config("mesh_single_config.json")
    args = parse_args(defaults)

    answer_path = args.answer_json or defaults["answer_json"].format(obj_name=args.obj_name)
    out_dir = args.output_dir or defaults["output_dir"].format(obj_name=args.obj_name)

    with open(answer_path, mode="r") as file:
        answer = json.load(file)

    craft = build_assembly(answer, args.img_name)
    print(craft.connections)

    tree, ordered_parts = craft.generate_tree()
    place_parts(craft, tree, ordered_parts)

    os.makedirs(out_dir, exist_ok=True)
    craft.save_parts(out_dir)
    print(f"Saved {len(craft.list_parts)} parts -> {out_dir}")


if __name__ == "__main__":
    main()
