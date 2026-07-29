"""Streamlined, class-based batch mesh generation pipeline.

Runtime settings default to ``config/mesh_gen_config.json`` and may be
overridden on the command line. All geometry logic lives in ``craft_assembly.py``.
"""
import os
import json
import copy
import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config_loader import load_config
from craft_assembly import (
    available_dims,
    CraftAssembly,
    normalize_response_text,
    init_records,
    collect_mod_names,
    get_part_names,
    get_opposite_face,
    apply_boolean_operations,
    update_pose_records,
    save_error_artifacts,
    rescale_parts,
    SURFACE_ALIGN_X, SURFACE_ALIGN_Y, SURFACE_ALIGN_Z,
    DRILL_ALIGN_X, DRILL_ALIGN_Y, DRILL_ALIGN_Z,
    CYLINDER_ORIENTATION,
)


@dataclass
class BatchConfig:
    input_jsonl: str
    obj_type: str
    target_function: str
    output_root: str
    error_root: str
    rescale_root: str = ""
    rescale_factor: float = 0.001
    ignore_ids: set = field(default_factory=set)


@dataclass
class BatchState:
    list_errors: list = field(default_factory=list)
    list_error_orientation: list = field(default_factory=list)
    list_error_missing_key: list = field(default_factory=list)
    list_error_wrong_name: list = field(default_factory=list)
    list_error_occupied: list = field(default_factory=list)
    error_list: list = field(default_factory=list)
    connection_list: list = field(default_factory=list)
    functional_list: list = field(default_factory=list)
    boolean_list: list = field(default_factory=list)
    root_list: list = field(default_factory=list)
    rotation_list: list = field(default_factory=list)
    positions_list: list = field(default_factory=list)
    saved_ids: list = field(default_factory=list)
    description_dict: dict = field(default_factory=dict)
    collision_same_place: int = 0
    error_count: int = 0

    def add_error(self, custom_id, reason, buckets=()):
        self.error_count += 1
        self.error_list.append({"custom_id": custom_id, "error_reason": reason})
        self.list_errors.append(custom_id)
        for bucket in buckets:
            bucket.append(custom_id)


class BatchMeshGenerator:
    def __init__(self, config):
        self.config = config
        self.state = BatchState()

    def run(self):
        all_answers = pd.read_json(self.config.input_jsonl, lines=True)
        for i in range(len(all_answers["custom_id"])):
            custom_id = all_answers["custom_id"][i]
            if custom_id in self.config.ignore_ids:
                continue
            print("------------------------------------------------------")
            print(custom_id)
            self._process_record(custom_id, all_answers["response"][i])
        self._rescale_outputs()
        self._finalize()

    def _process_record(self, custom_id, response_obj):
        connection_dict, functional_dict, boolean_dict, root_dict, rot_dict, pos_dict = init_records(custom_id)
        try:
            text_answer = response_obj["body"]["choices"][0]["message"]["content"]
            json_answer = json.loads(normalize_response_text(text_answer))
        except Exception as exc:
            self.state.add_error(custom_id, f"Failed to parse model response: {exc}", buckets=[self.state.list_error_missing_key])
            return

        craft_test = CraftAssembly(custom_id, available_dims, self.config.target_function)
        move_surface_list = []
        move_insert_list = []
        list_compare_corr = []
        mod_names = collect_mod_names(json_answer, json_answer.keys())
        collision_seen = False

        for part_name, part_spec in json_answer.items():
            ok, reason = self._validate_orientation(part_name, part_spec)
            if not ok:
                self.state.add_error(custom_id, reason, buckets=[self.state.list_error_orientation])
                return

            craft_test.generate_part(part_name, part_spec["AVAILABLE_OBJ"], part_spec["ORIENTATION"])
            ok, reason = self._register_modifications(craft_test, part_name, part_spec, mod_names)
            if not ok:
                self.state.add_error(custom_id, reason, buckets=[self.state.list_error_missing_key])
                return

            ok, reason, inserts, list_compare_corr, collision_seen = self._register_connections(
                custom_id=custom_id,
                part_name=part_name,
                part_spec=part_spec,
                connection_dict=connection_dict,
                move_surface_list=move_surface_list,
                move_insert_list=move_insert_list,
                list_compare_corr=list_compare_corr,
                collision_seen=collision_seen,
            )
            if not ok:
                self.state.add_error(custom_id, reason, buckets=[self.state.list_error_missing_key])
                return
            if inserts:
                pass

            exec_function = part_spec.get("exec_function", {}).get("value", False)
            if exec_function == "TRUE" or exec_function is True:
                functional_dict["parts"].append(part_name)

        if collision_seen:
            self.state.collision_same_place += 1

        list_parts_names = get_part_names(craft_test.parts)
        ok, reason, update_after = self._execute_insert_moves(custom_id, craft_test, move_insert_list, list_parts_names)
        if not ok:
            self.state.add_error(custom_id, reason, buckets=[self.state.list_error_wrong_name])
            return

        ok, reason, moved_parts = self._execute_surface_moves(custom_id, craft_test, move_surface_list, list_parts_names)
        if not ok:
            self.state.add_error(custom_id, reason, buckets=[self.state.list_error_occupied])
            return

        ok, reason = self._resolve_delayed_subassemblies(custom_id, craft_test, update_after, moved_parts)
        if not ok:
            self.state.add_error(custom_id, reason)
            return

        boolean_error, boolean_error_msg, floating_error, floating_list = apply_boolean_operations(craft_test, tol=0.5)
        if boolean_error or floating_error:
            reason = ""
            if boolean_error:
                reason += boolean_error_msg
            if floating_error:
                reason += f"The following pairs of parts do not have any overlap to create the slot/holes: {floating_list}"
            self.state.add_error(custom_id, reason)
            return

        out_dir = os.path.join(self.config.output_root, self.config.obj_type, custom_id)
        os.makedirs(out_dir, exist_ok=True)
        craft_test.save_parts(out_dir)
        self.state.saved_ids.append(custom_id)

        boolean_dict["pairs"] = craft_test.boolean_pairs
        update_pose_records(craft_test, connection_dict, functional_dict, root_dict, rot_dict, pos_dict)
        self.state.connection_list.append(connection_dict)
        self.state.functional_list.append(functional_dict)
        self.state.boolean_list.append(boolean_dict)
        self.state.root_list.append(root_dict)
        self.state.rotation_list.append(rot_dict)
        self.state.positions_list.append(pos_dict)
        self.state.description_dict[custom_id] = craft_test.description

    def _validate_orientation(self, part_name, part_spec):
        p_obj = part_spec["AVAILABLE_OBJ"]
        p_shape = available_dims[p_obj][-1]
        orientation = part_spec["ORIENTATION"]

        if p_shape == "cylinder" and orientation not in CYLINDER_ORIENTATION:
            return False, f"""The ORIENTATION of {part_name}: ({orientation}) is not valid. Should be one of ["FRONT_BACK", "LEFT_RIGHT", "TOP_BOTTOM"]."""

        if p_shape == "cuboid":
            dimensions_test = available_dims[p_obj][:3]
            final_orientation = copy.deepcopy(orientation)
            for d in dimensions_test:
                if d in final_orientation:
                    final_orientation.remove(d)
                else:
                    return False, f"""The ORIENTATION of {part_name}: ({orientation}) is not valid. The proposed valued {d} is not in the list of original dimensions {dimensions_test}"""
        return True, ""

    def _register_modifications(self, craft_test, part_name, part_spec, mod_names):
        for mod in part_spec.get("modifications", []):
            if "align_x" not in mod or "align_y" not in mod or "align_z" not in mod:
                reason = ""
                if "align_x" not in mod:
                    reason += f"""The modification ({mod}) field of {part_name} is missing the "align_x" key."""
                if "align_y" not in mod:
                    reason += f"""The modification ({mod}) field of {part_name} is missing the "align_y" key."""
                if "align_z" not in mod:
                    reason += f"""The modification ({mod}) field of {part_name} is missing the "align_z" key."""
                return False, reason

            if (mod["align_x"] not in SURFACE_ALIGN_X and mod["align_x"] not in mod_names) or \
               (mod["align_y"] not in SURFACE_ALIGN_Y and mod["align_y"] not in mod_names) or \
               (mod["align_z"] not in SURFACE_ALIGN_Z and mod["align_z"] not in mod_names):
                reason = ""
                if mod["align_x"] not in SURFACE_ALIGN_X and mod["align_x"] not in mod_names:
                    reason += f"""The align_x key for modification ({mod}) of {part_name} is not valid. {mod["align_x"]} not in ["FRONT", "CENTER", "BACK", "FRONT_BACK_FULL", "FRONT_BACK_HALF", "BACK_FRONT_HALF"]."""
                if mod["align_y"] not in SURFACE_ALIGN_Y and mod["align_y"] not in mod_names:
                    reason += f"""The align_y key for modification ({mod}) of {part_name} is not valid. {mod["align_y"]} not in ["RIGHT", "CENTER", "LEFT", "RIGHT_LEFT_FULL", "RIGHT_LEFT_HALF", "LEFT_RIGHT_HALF"]."""
                if mod["align_z"] not in SURFACE_ALIGN_Z and mod["align_z"] not in mod_names:
                    reason += f"""The align_z key for modification ({mod}) of {part_name} is not valid. {mod["align_z"]} not in ["HIGH", "CENTER", "LOW", "TOP_BOTTOM_FULL", "TOP_BOTTOM_HALF", "BOTTOM_TOP_HALF"]."""
                return False, reason

            if mod["align_x"] not in DRILL_ALIGN_X and mod["align_y"] not in DRILL_ALIGN_Y and mod["align_z"] not in DRILL_ALIGN_Z:
                return False, """One of the "align_x", "align_y" or "align_z" should be specifying the drilling direction and length, i.e., align_x should be in ["FRONT_BACK_FULL", "FRONT_BACK_HALF", "BACK_FRONT_HALF"] or align_y in ["RIGHT_LEFT_FULL", "RIGHT_LEFT_HALF", "LEFT_RIGHT_HALF"] or align_z in ["HIGH", "CENTER", "LOW", "TOP_BOTTOM_FULL", "TOP_BOTTOM_HALF", "BOTTOM_TOP_HALF"]."""

            craft_test.make_mod(part_name, mod["name"], mod["type"], mod["align_x"], mod["align_y"], mod["align_z"])
        return True, ""

    def _register_connections(
        self,
        custom_id,
        part_name,
        part_spec,
        connection_dict,
        move_surface_list,
        move_insert_list,
        list_compare_corr,
        collision_seen,
    ):
        all_conn_insert = []
        for conn in part_spec.get("connections", []):
            if "to_part" not in conn or "contact_type" not in conn:
                reason = ""
                if "to_part" not in conn:
                    reason += f"""The connection ({conn}) field of {part_name} is missing the "to_part" key."""
                if "contact_type" not in conn:
                    reason += f"""The connection ({conn}) field of {part_name} is missing the "contact_type" key."""
                return False, reason, None, list_compare_corr, collision_seen

            if conn["contact_type"] == "SURFACE":
                if "align_x" not in conn or "align_y" not in conn or "align_z" not in conn:
                    reason = ""
                    if "align_x" not in conn:
                        reason += f"""The connection ({conn}) field of {part_name} is missing the "align_x" key."""
                    if "align_y" not in conn:
                        reason += f"""The connection ({conn}) field of {part_name} is missing the "align_y" key."""
                    if "align_z" not in conn:
                        reason += f"""The connection ({conn}) field of {part_name} is missing the "align_z" key."""
                    return False, reason, None, list_compare_corr, collision_seen

                move_surface_list.append([part_name, conn["to_part"], conn["to_face"], conn["align_x"], conn["align_y"], conn["align_z"]])
                corr_key = [conn["to_part"], conn["to_face"], conn["align_x"], conn["align_y"], conn["align_z"]]
                if corr_key in list_compare_corr and not collision_seen:
                    collision_seen = True
                else:
                    list_compare_corr.append(corr_key)
            elif conn["contact_type"] == "INSERTED":
                all_conn_insert.append(conn["to_modification"])

            found_pair = False
            for c in connection_dict["connection_pairs"]:
                if part_name in c and conn["to_part"] in c:
                    found_pair = True
                    break
            if not found_pair:
                connection_dict["connection_pairs"].append([part_name, conn["to_part"]])
                if "type" in conn:
                    connection_dict["connection_type"].append(conn["type"])
                else:
                    return False, f"""The connection ({conn}) field of {part_name} is missing connection type.""", None, list_compare_corr, collision_seen

        if "modifications" not in part_spec and len(all_conn_insert) > 0:
            pair_check = [part_name, all_conn_insert]
            if pair_check not in move_insert_list:
                move_insert_list.append(pair_check)
        return True, "", all_conn_insert, list_compare_corr, collision_seen

    def _execute_insert_moves(self, custom_id, craft_test, move_insert_list, list_parts_names):
        update_after = []
        for part_name_1, modification_name in move_insert_list:
            if len(set(modification_name)) < len(modification_name):
                return False, "Some of the proposed modifications have the same name. They must have unique names.", None
            if part_name_1 not in list_parts_names:
                return False, f"{part_name_1} is not in the list of parts ({list_parts_names}).", None

            if len(modification_name) == 1:
                craft_test.move_insert(part_name_1, modification_name[0])
            else:
                reference_parts = craft_test.move_insert_multiple(part_name_1, modification_name)
                reference_parts.insert(0, part_name_1)
                update_after.append(reference_parts)
        return True, "", update_after

    def _execute_surface_moves(self, custom_id, craft_test, move_surface_list, list_parts_names):
        moved_parts = []
        pairs_moved = []
        occupied_slots = {}

        def _slot_key(part_name, face, ax, ay, az):
            return (part_name, face, ax, ay, az)

        for part_name_1, part_name_2, part_1_face, align_x, align_y, align_z in move_surface_list:
            if part_name_1 not in list_parts_names:
                return False, f"{part_name_1} is not in the list of parts ({list_parts_names}).", None
            if part_name_2 not in list_parts_names:
                return False, f"{part_name_2} is not in the list of parts ({list_parts_names}).", None
            if [part_name_2, part_name_1] in pairs_moved or [part_name_1, part_name_2] in pairs_moved:
                continue

            if part_name_2 not in moved_parts:
                error = craft_test.move_surface(part_name_1, part_name_2, part_1_face, align_x, align_y, align_z)
                moved_parts.append(part_name_2)
                pairs_moved.append([part_name_1, part_name_2])
                if error:
                    obstructing_part = occupied_slots.get(_slot_key(part_name_2, get_opposite_face(part_1_face), align_x, align_y, align_z))
                    if obstructing_part is not None:
                        return False, f"The placement of {part_name_2} to the {part_1_face} of {part_name_1} is not valid because {obstructing_part} is obstructing that location.", None
                    return False, f"The placement of {part_name_2} to the {part_1_face} of {part_name_1} is not valid.", None
                occupied_slots[_slot_key(part_name_1, part_1_face, align_x, align_y, align_z)] = part_name_2
                occupied_slots[_slot_key(part_name_2, get_opposite_face(part_1_face), align_x, align_y, align_z)] = part_name_1
            elif part_name_1 not in moved_parts:
                part_2_face = get_opposite_face(part_1_face)
                error = craft_test.move_surface(part_name_2, part_name_1, part_2_face, align_x, align_y, align_z)
                if error:
                    obstructing_part = occupied_slots.get(_slot_key(part_name_1, part_1_face, align_x, align_y, align_z))
                    if obstructing_part is not None:
                        return False, f"The placement of {part_name_2} to the {part_1_face} of {part_name_1} is not valid because {obstructing_part} is obstructing that location.", None
                    return False, f"The placement of {part_name_2} to the {part_1_face} of {part_name_1} is not valid.", None
                occupied_slots[_slot_key(part_name_2, part_2_face, align_x, align_y, align_z)] = part_name_1
                occupied_slots[_slot_key(part_name_1, part_1_face, align_x, align_y, align_z)] = part_name_2
                moved_parts.append(part_name_1)
                pairs_moved.append([part_name_2, part_name_1])
        return True, "", moved_parts

    def _resolve_delayed_subassemblies(self, custom_id, craft_test, update_after, moved_parts):
        status_all = [True for _ in update_after]
        if len(moved_parts) == 0 and len(craft_test.parts) > 0:
            moved_parts.append(craft_test.parts[0]["name"])

        loop_counter = -1
        previous_status = [True for _ in update_after]
        while any(status_all):
            if status_all == previous_status:
                loop_counter += 1
                if loop_counter > 10:
                    disconnected = [update_after[x] for x in range(len(status_all)) if not status_all[x]]
                    return False, f"This subassembly of parts is disconnected from the rest of the assembly: {disconnected}"
            previous_status = copy.deepcopy(status_all)

            for j in range(len(update_after)):
                if not status_all[j]:
                    continue
                move_3 = update_after[j]
                move_part = move_3[0]

                if move_part in moved_parts:
                    update_after = [[name, move_part] for name in move_3[1:] if name not in moved_parts]
                    status_all = [True for _ in update_after]
                    break

                ref_parts = move_3[1:]
                sum_pos = np.array([0.0, 0.0, 0.0])
                added = 0
                sub_parts = []
                for r_part in ref_parts:
                    if r_part not in moved_parts:
                        sub_parts.append(r_part)
                        continue
                    for part in craft_test.parts:
                        if part["name"] == r_part:
                            sum_pos += np.array(part["position"])
                            added += 1
                            break
                if added == 0:
                    continue

                sum_pos = sum_pos / added
                for part in craft_test.parts:
                    if part["name"] != move_part:
                        continue
                    part["position"] += sum_pos
                    mov_mesh = part["mesh"]
                    mov_mesh.apply_translation(sum_pos)
                    for sub in sub_parts:
                        for part2 in craft_test.parts:
                            if part2["name"] == sub:
                                part2["position"] = part["position"]
                                mov_mesh2 = part2["mesh"]
                                mov_mesh2.apply_translation(part["position"])
                                part2["mesh"] = mov_mesh2
                                break
                    part["mesh"] = mov_mesh
                    status_all[j] = False
                    moved_parts.append(move_part)
                    break
        return True, ""

    def _rescale_outputs(self):
        """Export a scaled copy of the crafts saved by this run."""
        if not self.config.rescale_root:
            return
        for custom_id in self.state.saved_ids:
            rescale_parts(
                os.path.join(self.config.output_root, self.config.obj_type, custom_id),
                os.path.join(self.config.rescale_root, self.config.obj_type, custom_id),
                self.config.rescale_factor,
            )

    def _finalize(self):
        print(len(self.state.error_list))
        print("Invalid orientation value:", len(self.state.list_error_orientation))
        print("Missing key for json format:", len(self.state.list_error_missing_key))
        print("Part name is not in list of parts:", len(self.state.list_error_wrong_name))
        print("Collision in the same place:", self.state.collision_same_place)
        print(self.state.list_error_occupied)

        error_files_dir = os.path.join(self.config.error_root, self.config.obj_type)
        save_error_artifacts(
            error_files_dir=error_files_dir,
            list_errors=self.state.list_errors,
            connection_list=self.state.connection_list,
            functional_list=self.state.functional_list,
            boolean_list=self.state.boolean_list,
            root_list=self.state.root_list,
            rotation_list=self.state.rotation_list,
            positions_list=self.state.positions_list,
            error_list=self.state.error_list,
            description_dict=self.state.description_dict,
        )


def parse_args(defaults):
    parser = argparse.ArgumentParser(description="Streamlined batch mesh generation pipeline.")
    parser.add_argument("--input-jsonl", default=defaults["input_jsonl"])
    parser.add_argument("--obj-type", default=defaults["obj_type"])
    parser.add_argument("--target-function", default=defaults["target_function"])
    parser.add_argument("--output-root", default=defaults["output_root"])
    parser.add_argument("--error-root", default=defaults["error_root"])
    parser.add_argument("--rescale-root", default=defaults.get("rescale_root", ""),
                        help="Root for the scaled copies of the meshes. Empty disables rescaling.")
    parser.add_argument("--rescale-factor", type=float, default=defaults.get("rescale_factor", 0.001))
    parser.add_argument("--ignore-id", action="append", default=list(defaults.get("ignore_ids", [])))
    return parser.parse_args()


def main():
    defaults = load_config("mesh_gen_config.json")
    args = parse_args(defaults)
    config = BatchConfig(
        input_jsonl=args.input_jsonl,
        obj_type=args.obj_type,
        target_function=args.target_function,
        output_root=args.output_root,
        error_root=args.error_root,
        rescale_root=args.rescale_root,
        rescale_factor=args.rescale_factor,
        ignore_ids=set(args.ignore_id),
    )
    BatchMeshGenerator(config).run()


if __name__ == "__main__":
    main()
