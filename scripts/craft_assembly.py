"""Shared geometry/assembly library for the mesh-generation pipeline.

Contains the ``CraftAssembly`` class plus the helper functions and constant
tables used by both ``gen_meshes.py`` and ``batch_mesh_gen_streamlined.py``.
The available-objects catalog is loaded from ``config/available_dims.json``.
"""
import trimesh
import numpy as np
import os
import json
from scipy.spatial.transform import Rotation as R
import copy

from config_loader import load_available_dims

FACE_OPPOSITE = {
    "TOP": "BOTTOM",
    "BOTTOM": "TOP",
    "RIGHT": "LEFT",
    "LEFT": "RIGHT",
    "FRONT": "BACK",
    "BACK": "FRONT",
}

# Euler angles and the permutation that maps the base dimensions
# (diameter, diameter, height) onto the world axes after the rotation.
CYLINDER_ROTATION = {
    "LEFT_RIGHT": ([90, 0, 0], [0, 2, 1]),
    "FRONT_BACK": ([0, 90, 0], [2, 0, 1]),
    "TOP_BOTTOM": ([0, 0, 0], [0, 1, 2]),
}

CUBOID_EULER_BY_PERM = {
    (0, 1, 2): [0, 0, 0],
    (1, 0, 2): [0, 0, 90],
    (2, 1, 0): [0, 90, 0],
    (0, 2, 1): [90, 0, 0],
    (1, 2, 0): [90, 0, 90],
    (2, 0, 1): [0, 90, 90],
}

INSERT_ALIGN_GROUPS = {
    "x": {"FRONT", "BACK", "CENTER", "FRONT_BACK_FULL", "FRONT_BACK_HALF", "BACK_FRONT_HALF"},
    "y": {"RIGHT", "LEFT", "CENTER", "RIGHT_LEFT_FULL", "RIGHT_LEFT_HALF", "LEFT_RIGHT_HALF"},
    "z": {"HIGH", "LOW", "CENTER", "TOP_BOTTOM_FULL", "TOP_BOTTOM_HALF", "BOTTOM_TOP_HALF"},
}

available_dims = load_available_dims()



def get_opposite_face(face):
    if face not in FACE_OPPOSITE:
        raise ValueError(f"Unknown face value: {face}")
    return FACE_OPPOSITE[face]


def normalize_response_text(text):
    text = text.replace("AVAILABLE OBJ", "AVAILABLE_OBJ")
    text = text.replace("available_obj", "AVAILABLE_OBJ")
    text = text.replace("orientation", "ORIENTATION")
    return text.replace("-", "_")


def to_homogeneous(rotation_matrix):
    tf = np.eye(4)
    tf[:3, :3] = rotation_matrix
    return tf


def build_cylinder_by_orientation(dimensions, orientation):
    if orientation not in CYLINDER_ROTATION:
        raise ValueError(f"Invalid cylinder orientation: {orientation}")
    rot_euler, axis_order = CYLINDER_ROTATION[orientation]
    part = trimesh.creation.cylinder(radius=dimensions[0] / 2, height=dimensions[2], sections=200)
    rotation = R.from_euler("xyz", rot_euler, degrees=True).as_matrix()
    part.apply_transform(to_homogeneous(rotation))
    dimensions_tf = [dimensions[axis_order[0]], dimensions[axis_order[1]], dimensions[axis_order[2]]]
    return part, dimensions_tf, rot_euler


def aabb_separated(bounds_a, bounds_b, tol=0.5):
    """Whether two axis-aligned bounding boxes are further apart than ``tol``.

    Two parts that a plan says are connected must at least touch. When this
    returns True for such a pair, the part is floating.
    """
    a_half = (bounds_a[1] - bounds_a[0]) / 2.0
    b_half = (bounds_b[1] - bounds_b[0]) / 2.0
    a_center = (bounds_a[1] + bounds_a[0]) / 2.0
    b_center = (bounds_b[1] + bounds_b[0]) / 2.0
    return bool(np.any(np.abs(b_center - a_center) > (a_half + b_half) + tol))


def lookup_record(frame, custom_id, *columns):
    """The named columns of the row for ``custom_id``, or ``None`` if absent.

    Reads the per-craft tables ``save_error_artifacts`` writes, which are keyed
    by ``custom_id`` rather than indexed.
    """
    for i in range(len(frame["custom_id"])):
        if frame["custom_id"][i] == custom_id:
            return tuple(frame[column][i] for column in columns)
    return None


def find_cuboid_rotation_euler(dimensions, orientation):
    for perm in ([0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]):
        if [dimensions[i] for i in perm] == list(orientation):
            return CUBOID_EULER_BY_PERM.get(tuple(perm), [0, 0, 0])
    return [0, 0, 0]


def write_jsonl(file_path, records):
    with open(file_path, "w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def save_error_artifacts(
    error_files_dir,
    list_errors,
    connection_list,
    functional_list,
    boolean_list,
    root_list,
    rotation_list,
    positions_list,
    error_list,
    description_dict,
):
    os.makedirs(error_files_dir, exist_ok=True)

    np.save(f"{error_files_dir}/format.npy", list_errors)
    write_jsonl(f"{error_files_dir}/connections.jsonl", connection_list)
    write_jsonl(f"{error_files_dir}/functional.jsonl", functional_list)
    write_jsonl(f"{error_files_dir}/boolean.jsonl", boolean_list)
    write_jsonl(f"{error_files_dir}/root.jsonl", root_list)
    write_jsonl(f"{error_files_dir}/rotation.jsonl", rotation_list)
    write_jsonl(f"{error_files_dir}/position.jsonl", positions_list)
    write_jsonl(f"{error_files_dir}/error.jsonl", error_list)

    with open(f"{error_files_dir}/description.json", "w") as file:
        file.write(json.dumps(description_dict))


def rescale_parts(source_dir, target_dir, scale):
    """Write a scaled copy of every mesh in ``source_dir`` into ``target_dir``.

    The pipeline builds parts in millimetres; downstream consumers expect
    metres, so the meshes are re-exported scaled by ``scale``.
    """
    if not os.path.isdir(source_dir):
        return
    os.makedirs(target_dir, exist_ok=True)
    for mesh_name in sorted(os.listdir(source_dir)):
        mesh = trimesh.load(os.path.join(source_dir, mesh_name))
        mesh.apply_scale(scale)
        mesh.export(os.path.join(target_dir, mesh_name))
    print(f"Saved rescaled parts to {target_dir}")


def init_records(custom_id):
    connection_dict = {
        "custom_id": custom_id,
        "connection_pairs": [],
        "connection_type": [],
    }
    functional_dict = {"custom_id": custom_id, "parts": []}
    boolean_dict = {"custom_id": custom_id}
    root_dict = {"custom_id": custom_id}
    rot_dict = {"custom_id": custom_id, "parts": [], "values": []}
    pos_dict = {"custom_id": custom_id, "parts": [], "values": []}
    return connection_dict, functional_dict, boolean_dict, root_dict, rot_dict, pos_dict


def collect_mod_names(json_answer, list_parts):
    mod_names = []
    for part in list_parts:
        for mod in json_answer[part].get("modifications", []):
            mod_names.append(mod["name"])
    return mod_names


def get_part_names(parts):
    return [part["name"] for part in parts]


def apply_boolean_operations(craft_test, tol=0.5):
    floating_error = False
    floating_list = []
    boolean_error = False
    boolean_error_msg = ""
    parts_by_name = {part["name"]: part for part in craft_test.parts}

    for b_mod, b_1 in craft_test.boolean_pairs:
        b_mod_mesh = parts_by_name[b_mod]["mesh"]
        b_1_mesh = parts_by_name[b_1]["mesh"]

        b_1_pos = b_1_mesh.centroid
        b_1_mesh_copy = copy.deepcopy(b_1_mesh)
        b_1_mesh.apply_translation(-b_1_mesh.centroid)
        b_1_mesh.apply_scale(1.15)
        b_1_mesh.apply_translation(b_1_pos)
        b_mod_mesh_bool = trimesh.boolean.difference((b_mod_mesh, b_1_mesh), check_volume=False)

        if len(b_mod_mesh_bool.vertices) == 0:
            boolean_error = True
            boolean_error_msg += f"{b_mod} was completely erased during the boolean operation. "
            continue

        if aabb_separated(b_mod_mesh_bool.bounds, b_1_mesh.bounds, tol):
            floating_error = True
            floating_list.append([b_mod, b_1])

        parts_by_name[b_mod]["mesh"] = b_mod_mesh_bool
        parts_by_name[b_1]["mesh"] = b_1_mesh_copy

    return boolean_error, boolean_error_msg, floating_error, floating_list


def update_pose_records(craft_test, connection_dict, functional_dict, root_dict, rot_dict, pos_dict):
    count_parts = [0 for _ in craft_test.parts]
    for n, part in enumerate(craft_test.parts):
        for c_pair in connection_dict["connection_pairs"]:
            if part["name"] in c_pair:
                count_parts[n] += 1
        if part["name"] in functional_dict["parts"]:
            count_parts[n] = 0
    root_idx = int(np.argmax(count_parts))
    root_dict["part"] = craft_test.parts[root_idx]["name"]

    for part in craft_test.parts:
        rot_dict["parts"].append(part["name"])
        pos_dict["parts"].append(part["name"])
        rot_dict["values"].append(part["rotation_euler"])
        pos_dict["values"].append(list(np.array(part["position"]).astype(float)))


def update_sub(sub_p, offset):
    mask = np.array(sub_p["update"], dtype=float)
    offset = np.array(offset, dtype=float) * mask
    sub_p["position"] = list(np.array(sub_p["position"], dtype=float) + offset)
    sub_p["mesh"].apply_translation(offset)
    sub_p["updated_sub"] = True
    for child in sub_p["sub_parts"]:
        if not child["updated_sub"]:
            update_sub(child, offset)

class CraftAssembly:
    def __init__(self, img_name, available_objs, target_function):
        self.img_name = img_name
        self.available_objs = available_objs
        self.target_function = target_function

        self.parts = []
        self.modifications = []
        self.connection_pairs = []
        self.boolean_pairs = []
        self.description = ""

    def _parts_by_name(self):
        return {part["name"]: part for part in self.parts}

    def _mods_by_name(self):
        return {mod["name"]: mod for mod in self.modifications}

    def _build_part_mesh(self, dimensions, shape, orientation):
        if shape == "cuboid":
            part = trimesh.creation.box(orientation)
            return part, list(orientation), find_cuboid_rotation_euler(dimensions[:3], orientation)
        if shape == "cylinder":
            return build_cylinder_by_orientation(dimensions, orientation)
        raise ValueError(f"Unknown shape type: {shape}")

    def _surface_base_position(self, part_1_pos, part_1_dims, part_2_dims, part_1_face):
        match part_1_face:
            case "TOP":
                return [part_1_pos[0], part_1_pos[1], part_1_pos[2] + part_1_dims[2] / 2 + part_2_dims[2] / 2], [part_1_dims[0], part_1_dims[1], 0], "BOTTOM"
            case "BOTTOM":
                return [part_1_pos[0], part_1_pos[1], part_1_pos[2] - part_1_dims[2] / 2 - part_2_dims[2] / 2], [part_1_dims[0], part_1_dims[1], 0], "TOP"
            case "RIGHT":
                return [part_1_pos[0], part_1_pos[1] + part_1_dims[1] / 2 + part_2_dims[1] / 2, part_1_pos[2]], [part_1_dims[0], 0, part_1_dims[2]], "LEFT"
            case "LEFT":
                return [part_1_pos[0], part_1_pos[1] - part_1_dims[1] / 2 - part_2_dims[1] / 2, part_1_pos[2]], [part_1_dims[0], 0, part_1_dims[2]], "RIGHT"
            case "FRONT":
                return [part_1_pos[0] + part_1_dims[0] / 2 + part_2_dims[0] / 2, part_1_pos[1], part_1_pos[2]], [0, part_1_dims[1], part_1_dims[2]], "BACK"
            case "BACK":
                return [part_1_pos[0] - part_1_dims[0] / 2 - part_2_dims[0] / 2, part_1_pos[1], part_1_pos[2]], [0, part_1_dims[1], part_1_dims[2]], "FRONT"
        raise ValueError(f"Unknown face value: {part_1_face}")

    def _apply_surface_alignment(self, part_2_pos, part_1_dims, part_2_dims, align_x, align_y, align_z, offset, highest_dim):
        if align_x == "FRONT":
            part_2_pos[0] += part_1_dims[0] / 2 - part_2_dims[0] / 2
            if offset and highest_dim == 0:
                part_2_pos[0] -= part_1_dims[0] / 4
        elif align_x == "BACK":
            part_2_pos[0] -= part_1_dims[0] / 2 - part_2_dims[0] / 2
            if offset and highest_dim == 0:
                part_2_pos[0] += part_1_dims[0] / 4
        elif align_x == "CENTER" and offset and highest_dim == 0:
            part_2_pos[0] -= part_1_dims[0] / 4

        if align_y == "RIGHT":
            part_2_pos[1] += part_1_dims[1] / 2 - part_2_dims[1] / 2
            if offset and highest_dim == 1:
                part_2_pos[1] -= part_1_dims[1] / 4
        elif align_y == "LEFT":
            part_2_pos[1] -= part_1_dims[1] / 2 - part_2_dims[1] / 2
            if offset and highest_dim == 1:
                part_2_pos[1] += part_1_dims[1] / 4
        elif align_y == "CENTER" and offset and highest_dim == 1:
            part_2_pos[1] -= part_1_dims[1] / 4

        if align_z == "HIGH":
            part_2_pos[2] += part_1_dims[2] / 2 - part_2_dims[2] / 2
            if offset and highest_dim == 2:
                part_2_pos[2] -= part_1_dims[2] / 4
        elif align_z == "LOW":
            part_2_pos[2] -= part_1_dims[2] / 2 - part_2_dims[2] / 2
            if offset and highest_dim == 2:
                part_2_pos[2] += part_1_dims[2] / 4
        elif align_z == "CENTER" and offset and highest_dim == 2:
            part_2_pos[2] -= part_1_dims[2] / 4
        return part_2_pos

    def _resolve_insert_axis(self, axis_align, axis_dim, part_mod_pos_val, part_1_dim, offset, highest_dim, axis_index, mods_by_name):
        if axis_align == "CENTER":
            return part_mod_pos_val, True
        if axis_align in ("FRONT", "RIGHT", "HIGH"):
            value = part_mod_pos_val + axis_dim / 2 - 1.2 * part_1_dim / 2
            if offset and highest_dim == axis_index:
                value -= axis_dim / 4
            return value, True
        if axis_align in ("BACK", "LEFT", "LOW"):
            value = part_mod_pos_val - axis_dim / 2 + 1.2 * part_1_dim / 2
            if offset and highest_dim == axis_index:
                value += axis_dim / 4
            return value, True
        if axis_align in ("FRONT_BACK_HALF", "RIGHT_LEFT_HALF", "TOP_BOTTOM_HALF"):
            value = part_mod_pos_val + part_1_dim / 2
            if offset and highest_dim == axis_index:
                value -= axis_dim / 8
            return value, True
        if axis_align in ("BACK_FRONT_HALF", "LEFT_RIGHT_HALF", "BOTTOM_TOP_HALF"):
            value = part_mod_pos_val - part_1_dim / 2
            if offset and highest_dim == axis_index:
                value += axis_dim / 8
            return value, True
        if axis_align in ("FRONT_BACK_FULL", "RIGHT_LEFT_FULL", "TOP_BOTTOM_FULL"):
            return part_mod_pos_val, True
        ref_mod = mods_by_name.get(axis_align)
        if ref_mod:
            return ref_mod["position"][axis_index], False
        return part_mod_pos_val, True

    def _compute_insert_position(self, part_mod_mesh, part_mod_dims, part_1_dims, mod_align, part_mod_occ):
        location = ["modification", mod_align[0], mod_align[1], mod_align[2]]
        offset = location in part_mod_occ
        highest_dim = int(np.argmax(part_mod_dims))
        part_update = [True, True, True]
        part_mod_pos = list(part_mod_mesh.centroid)
        mods_by_name = self._mods_by_name()

        part_mod_pos[0], part_update[0] = self._resolve_insert_axis(
            mod_align[0], part_mod_dims[0], part_mod_pos[0], part_1_dims[0], offset, highest_dim, 0, mods_by_name
        )
        part_mod_pos[1], part_update[1] = self._resolve_insert_axis(
            mod_align[1], part_mod_dims[1], part_mod_pos[1], part_1_dims[1], offset, highest_dim, 1, mods_by_name
        )
        part_mod_pos[2], part_update[2] = self._resolve_insert_axis(
            mod_align[2], part_mod_dims[2], part_mod_pos[2], part_1_dims[2], offset, highest_dim, 2, mods_by_name
        )
        return location, part_mod_pos, part_update

    def _place_part_mesh_at(self, part_mesh, position):
        moved = copy.deepcopy(part_mesh)
        moved.apply_translation(-moved.centroid)
        moved.apply_translation(position)
        return moved

    def generate_part(self, name, available_obj, orientation):
        """Generate a part with the given name and dimensions."""
        if available_obj not in self.available_objs:
            raise ValueError(f"Available object {available_obj} not found.")
        dimensions = self.available_objs[available_obj]
        part, dimensions_tf, rot_euler = self._build_part_mesh(dimensions, dimensions[3], orientation)
        part.apply_translation(-part.centroid)

        if len(self.parts) == 0:
            self.description += f"The part {name} is initially placed. \n"

        self.parts.append({
            "name": name,
            "available_obj": available_obj,
            "mesh": part,
            "dimensions": dimensions_tf,
            "position": [0, 0, 0],
            "rotation": orientation,
            "shape": dimensions[3],
            "occupied": [],
            "direction": "",
            "rotation_euler": rot_euler,
            "sub_parts": [],
            "update": [True, True, True],
            "already_sub": False,
            "updated_sub": False,
        })
        print(f"Generated part: {name} with dimensions {dimensions[:3]} and shape {dimensions[3]}")
        
    def move_surface(self, part_name_1, part_name_2, part_1_face, align_x, align_y, align_z):
        """Move the part to the specified position."""
        parts_by_name = self._parts_by_name()
        part_1 = parts_by_name[part_name_1]
        part_2 = parts_by_name[part_name_2]
        part_1_pos = part_1["position"]
        part_1_dims = part_1["dimensions"]
        part_1_occ = part_1["occupied"]
        part_1_sub = part_1["sub_parts"]
        part_2_dims = part_2["dimensions"]
        part_2_sub = part_2["sub_parts"]
        part_2_pos_init = part_2["position"]
        part_2_occ = part_2["occupied"]

        location = [part_1_face, align_x, align_y, align_z]
        part_2_face = get_opposite_face(part_1_face)
        location_2 = [part_2_face, align_x, align_y, align_z]

        offset = location in part_1_occ
        if location_2 in part_2_occ:
            return True

        part_2_pos, mask_dims, part_2_dir = self._surface_base_position(part_1_pos, part_1_dims, part_2_dims, part_1_face)
        highest_dim = int(np.argmax(mask_dims))
        part_2_pos = self._apply_surface_alignment(part_2_pos, part_1_dims, part_2_dims, align_x, align_y, align_z, offset, highest_dim)

        if location not in part_1_occ:
            part_1_occ.append(location)
        if location_2 not in part_2_occ:
            part_2_occ.append(location_2)
        full_offset = np.array(part_2_pos) - np.array(part_2_pos_init)

        part_1_sub.append(part_2)

        self.description += f"The part {part_name_2} is placed to the {part_1_face} of {part_name_1}."
        if align_x != "CONTACT":
            if align_x == "FRONT":
                self.description += f"It is shifted in the X-axis so that the edges at the front are aligned."
            elif align_x == "BACK":
                self.description += f"It is shifted in the X-axis so that the edges at the back are aligned."
            elif align_x == "CENTER":
                self.description += "It is centered in the X-axis."
        if align_y != "CONTACT":
            if align_y == "RIGHT":
                self.description += f"It is shifted in the Y-axis so that the edges at the right side are aligned."
            elif align_y == "LEFT":
                self.description += f"It is shifted in the Y-axis so that the edges at the left side are aligned."
            elif align_y == "CENTER":
                self.description += "It is centered in the Y-axis."
        if align_z != "CONTACT":
            if align_z == "HIGH":
                self.description += f"It is shifted in the Z-axis so that the edges at the top are aligned."
            elif align_z == "LOW":
                self.description += f"It is shifted in the Z-axis so that the edges at the bottom are aligned."
            elif align_z == "CENTER":
                self.description += "It is centered in the Z-axis."

        self.description += "\n"

        part_2["position"] = part_2_pos
        part_2["mesh"].apply_translation(-part_2["mesh"].centroid)
        part_2["mesh"].apply_translation(np.array(part_2_pos))
        part_2["direction"] = part_2_dir
        print(f"Moved {part_name_2} to position {part_2_pos}")
        for sub_p in part_2_sub:
            update_sub(sub_p, full_offset)

    
    def move_insert(self, part_name_1, modification_name):
        part_positions, mod_parts = self._process_insert_modifications(part_name_1, [modification_name], align_to_average=False)
        _ = part_positions
        return mod_parts
    
    def move_insert_multiple(self, part_name_1, modification_name_list):
        _, mod_parts = self._process_insert_modifications(part_name_1, modification_name_list, align_to_average=True)
        return mod_parts

    def _process_insert_modifications(self, part_name_1, modification_names, align_to_average):
        parts_by_name = self._parts_by_name()
        mods_by_name = self._mods_by_name()
        part_1 = parts_by_name[part_name_1]

        part_mod_positions = []
        mod_parts = []
        last_part_update = [True, True, True]

        for modification_name in modification_names:
            mod = mods_by_name[modification_name]
            mod_name = mod["name"]
            mod_part = mod["part_name"]
            mod_align = mod["alignment"]
            mod_parts.append(mod_part)

            # An over-specified hole names several drilling directions. If the
            # part already lies along one of them, honour that rather than the
            # X-then-Y-then-Z default, so a plan that contradicts itself does
            # not silently rotate the part away from its stated ORIENTATION.
            candidates = mod["orientations"] or [mod["orientation"]]
            if (part_1["shape"] == "cylinder" and len(candidates) > 1
                    and part_1["rotation"] in candidates
                    and part_1["rotation"] != mod["orientation"]):
                print(f"The modification {mod['name']} specifies several drilling directions "
                      f"{candidates}; keeping {part_name_1} at its own orientation "
                      f"{part_1['rotation']}.")
                mod["orientation"] = part_1["rotation"]

            if part_1["shape"] == "cylinder" and part_1["rotation"] != mod["orientation"]:
                print(f"The orientation of {part_1['rotation']} doesn't match the orientation of the hole {mod['orientation']}.")
                print("Adjusting the cylinder.")
                part_dims = available_dims[part_1["available_obj"]][:3]
                part_adj, dimensions_tf, rot_euler = build_cylinder_by_orientation(part_dims, mod["orientation"])
                part_adj.apply_translation(-part_adj.centroid)
                part_1["dimensions"] = dimensions_tf
                part_1["rotation"] = mod["orientation"]
                part_1["mesh"] = part_adj
                part_1["rotation_euler"] = rot_euler

            mod_parent = parts_by_name[mod_part]
            location, part_mod_pos, part_update = self._compute_insert_position(
                mod_parent["mesh"],
                mod_parent["dimensions"],
                part_1["dimensions"],
                mod_align,
                mod_parent["occupied"],
            )
            last_part_update = part_update

            if location not in mod_parent["occupied"]:
                mod_parent["occupied"].append(location)

            part_mod_positions.append(part_mod_pos)
            mod["position"] = part_mod_pos

            print(f"{mod_name} positioned at {part_mod_pos}")
            self.boolean_pairs.append([mod_part, part_name_1])
            print(f"Boolean pair added: {mod_part} and {part_name_1}")
            self.description += f"The {mod_name} is made to {mod_part} at {mod_align}. The {part_name_1} is inserted at {mod_name}. \n"

        target_pos = part_mod_positions[0]
        if align_to_average and part_mod_positions:
            target_pos = list(np.mean(np.array(part_mod_positions, dtype=float), axis=0))

        part_1["mesh"] = self._place_part_mesh_at(part_1["mesh"], target_pos)
        part_1["position"] = target_pos
        part_1["update"] = last_part_update

        if not align_to_average and mod_parts:
            mod_parent = parts_by_name[mod_parts[0]]
            if not part_1["already_sub"]:
                mod_parent["sub_parts"].append(part_1)
                part_1["already_sub"] = True
        return part_mod_positions, mod_parts

    def make_mod(self, part_name, mod_name, mod_type, align_x, align_y, align_z):

        new_mod = {
            "name": mod_name,
            "mod_type": mod_type,
            "position": [],
            "orientation": "",
            "orientations": [],
            "part_name": part_name,
            "alignment": [align_x, align_y, align_z]
        }

        # A plan may name a drilling direction on more than one axis. Keep every
        # candidate so the part being inserted can pick the one it already lies
        # along; ``orientation`` stays the X-then-Y-then-Z pick as the default.
        if align_x in INSERT_ALIGN_GROUPS["x"] - {"FRONT", "BACK", "CENTER"}:
            new_mod["orientations"].append("FRONT_BACK")
        if align_y in INSERT_ALIGN_GROUPS["y"] - {"RIGHT", "LEFT", "CENTER"}:
            new_mod["orientations"].append("LEFT_RIGHT")
        if align_z in INSERT_ALIGN_GROUPS["z"] - {"HIGH", "LOW", "CENTER"}:
            new_mod["orientations"].append("TOP_BOTTOM")

        if new_mod["orientations"]:
            new_mod["orientation"] = new_mod["orientations"][0]

        self.modifications.append(new_mod)

    def evaluate_functional_part(self, part_name, target_function):
        """Evaluate if the part can perform the target function."""
        print(f"Evaluating if {part_name} can perform the target function: {target_function}")

    def discard(self, part_name):
        """Discard the part, it will not be used in the final assembly."""
        for part in self.parts:
            if part["name"] == part_name:
                self.parts.remove(part)
                print(f"Discarded part: {part_name}")
                return
        raise ValueError(f"Part {part_name} not found.")

    def save_parts(self, output_dir):
        """Save the parts to the specified directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for part in self.parts:
            part_mesh = part["mesh"]
            part_name = part["name"].replace("/", "")
            part_mesh.export(os.path.join(output_dir, f"{part_name}.obj"))
            print(f"Saved part: {part_name} to {output_dir}")

# Valid alignment tokens, grouped by the operation that consumes them.
SURFACE_ALIGN_X = {"FRONT", "CENTER", "BACK", "FRONT_BACK_FULL", "FRONT_BACK_HALF", "BACK_FRONT_HALF"}
SURFACE_ALIGN_Y = {"RIGHT", "CENTER", "LEFT", "RIGHT_LEFT_FULL", "RIGHT_LEFT_HALF", "LEFT_RIGHT_HALF"}
SURFACE_ALIGN_Z = {"HIGH", "CENTER", "LOW", "TOP_BOTTOM_FULL", "TOP_BOTTOM_HALF", "BOTTOM_TOP_HALF"}
DRILL_ALIGN_X = {"FRONT_BACK_FULL", "FRONT_BACK_HALF", "BACK_FRONT_HALF"}
DRILL_ALIGN_Y = {"RIGHT_LEFT_FULL", "RIGHT_LEFT_HALF", "LEFT_RIGHT_HALF"}
DRILL_ALIGN_Z = {"TOP_BOTTOM_FULL", "TOP_BOTTOM_HALF", "BOTTOM_TOP_HALF"}
CYLINDER_ORIENTATION = {"FRONT_BACK", "LEFT_RIGHT", "TOP_BOTTOM"}
