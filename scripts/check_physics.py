"""Run every generated craft through Isaac Sim and record which ones work.

Takes the meshes and pose records written by the mesh-generation stage, spawns
each craft as a set of rigid bodies, and applies the force its target function
calls for -- pushing a truck forward, pressing down on a chair, driving a hammer
into a nail. A craft passes only if it does the job without touching the ground
where it should not, and without shedding parts.

Failures are written to ``<results>/<out_dir>/error/<obj_type>/physics.jsonl``,
one ``{custom_id: [{failure_reason}, {additional_info}]}`` record per line --
the file the ``reprompt_physics.py`` feedback is built from.

Runtime settings default to ``config/check_physics_config.json`` and may be
overridden on the command line. Note that the simulator has to be started before
the Omniverse modules can be imported, so the usual import block is split in two.

Usage:
    python check_physics.py
    python check_physics.py --obj-type truck --headless
"""
import argparse
import asyncio
import glob
import json
import os

from config_loader import load_config


def parse_args(defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj-type", default=defaults["obj_type"])
    parser.add_argument("--in-dir", default=defaults["in_dir"],
                        help="round the meshes and pose records are read from")
    parser.add_argument("--out-dir", default=defaults["out_dir"],
                        help="round the physics results are written to")
    parser.add_argument("--results-root", default=defaults["results_root"])
    parser.add_argument("--utils-root", default=defaults["utils_root"],
                        help="folder holding the nail/board meshes used by the hit test")
    parser.add_argument("--start-index", type=int, default=defaults["start_index"])
    parser.add_argument("--max-files", type=int, default=defaults["max_files"])
    parser.add_argument("--density", type=float, default=defaults["density"])
    parser.add_argument("--headless", action="store_true", default=defaults["headless"])
    parser.add_argument("--use-sdf", action="store_true", default=defaults["use_sdf"],
                        help="reduce the SDF triangle count of every collider")
    return parser.parse_args()


CONFIG = load_config("check_physics_config.json")
ARGS = parse_args(CONFIG)

### Initialize physics simulator
# The simulator must exist before any of the isaacsim.core/omni extension
# modules can be imported, so these three lines have to run at import time,
# ahead of the rest of the imports.
import isaacsim  # noqa: E402,F401
import omni  # noqa: E402
from isaacsim import SimulationApp  # noqa: E402

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
simulation_app = SimulationApp({"headless": ARGS.headless})

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import pandas as pd  # noqa: E402

# The omni.isaac.* extensions were renamed to isaacsim.* in Isaac Sim 4.5 and
# removed in 5.0. In the same rename the single-prim classes gained a "Single"
# prefix, because the old plural-prim *View classes took over the bare names.
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
import isaacsim.core.utils.prims as prims_utils  # noqa: E402
from isaacsim.core.prims import SingleGeometryPrim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.core.api.materials import PhysicsMaterial  # noqa: E402

import omni.client  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402

from pxr import UsdPhysics, PhysxSchema, Gf, Usd, UsdGeom, Sdf, PhysicsSchemaTools  # noqa: E402
from typing import Union  # noqa: E402

from omni.physx import get_physx_simulation_interface  # noqa: E402

from scipy.spatial.transform import Rotation as R  # noqa: E402

enable_extension("omni.kit.asset_converter")

# ImageNet synset prefixes, which is all that identifies an object's class once
# the plans have been flattened into one folder. Truck is the fallback.
CLASS_PREFIXES = {
    "hammer": ("n03481172_",),
    "bookshelf": ("n02871439_", "n04190052_"),
    "skateboard": ("n04225987_",),
    "scooter": ("n04149374_", "pexels"),
    "bus": ("n02924116_",),
    "chair": ("n03001627_",),
    "table": ("n03063968_",),
}

CLASS_FUNCTION = {
    "hammer": "hit",
    "bookshelf": "support",
    "chair": "support",
    "table": "support",
    "skateboard": "rolling",
    "scooter": "rolling",
    "bus": "rolling",
    "truck": "rolling",
}

GROUND_PLANE_PATH = "/World/defaultGroundPlane/GroundPlane/CollisionPlane"
PHYSICS_MATERIAL_PATH = "/World/Physics_Materials/physics_material"
FINAL_STEP = 1500
STABILITY_FINAL_STEP = 5000

# A repeated contact only counts as the craft dropping onto the ground once it
# has persisted this many reports; a single frame is usually simulator noise.
GROUND_CONTACT_PATIENCE = 100


def classify(file_name):
    """The object class of a plan id, from its ImageNet prefix."""
    for name, prefixes in CLASS_PREFIXES.items():
        if any(prefix in file_name for prefix in prefixes):
            return name
    return "truck"


def prim_name(part_name):
    """A part name as it appears in the USD stage: no spaces, brackets or dashes."""
    return part_name.replace(" ", "").replace("(", "").replace(")", "").replace("-", "")


def part_path(part_name):
    return "/World/parts/{}".format(prim_name(part_name))


def distance(pos_a, pos_b):
    return float(np.linalg.norm(np.array(pos_a) - np.array(pos_b)))


### Functions for physics simulator
async def convert(in_file, out_file, load_materials=False):
    # This import causes conflicts when global
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # setup converter and flags
    converter_context.ignore_materials = not load_materials
    # The meshes are authored in metres and Z-up. Left to its defaults the
    # converter tags the USD as centimetres and Y-up, and while the render
    # geometry still measures up, PhysX cooks the collider 100x too small --
    # every part then drops through to the ground plane as a speck.
    converter_context.use_meter_as_world_unit = True
    converter_context.convert_stage_up_z = True
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, progress_callback, converter_context)
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success
#--------------------------------------------------------------------#
def convert_to_usd(input_model_path, converted_model_path):
    """Convert one mesh to USD, unless it has been converted already."""
    if not os.path.exists(converted_model_path):
        status = asyncio.get_event_loop().run_until_complete(
            convert(input_model_path, converted_model_path, True)
        )
        if not status:
            print(f"ERROR Status is {status}")
    print(f"---Added {converted_model_path}")
#--------------------------------------------------------------------#
def asset_convert(in_folder, out_folder):
    supported_file_formats = ["stl", "obj", "fbx"]
    (result, models) = omni.client.list(in_folder)
    for i, entry in enumerate(models):
        model = str(entry.relative_path)
        model_name = os.path.splitext(model)[0]
        model_format = (os.path.splitext(model)[1])[1:]
        # Supported input file formats
        if model_format in supported_file_formats:
            convert_to_usd(in_folder + "/" + model,
                           out_folder + "/" + model_name.replace(" ", "_") + ".usd")
#--------------------------------------------------------------------#
def get_prim_by_path(stage: Usd.Stage, prim_path: Union[str, Sdf.Path]) -> Usd.Prim:
    return stage.GetPrimAtPath(prim_path)
#--------------------------------------------------------------------#
def add_part(part_name, translation, rotation, scene, stage, added_parts, volume, density, folder_name, scale=np.array([0.001, 0.001, 0.001]), sdf=True):
    root_prim_path = os.path.join(folder_name, "assets")
    part_name = prim_name(part_name)
    prim_path = "/World/parts/{}".format(part_name)
    usd_path = root_prim_path + "/{}.usd".format(part_name)
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    added_parts.append(
        scene.add(
            SingleRigidPrim(
                prim_path = prim_path,
                name = "{}".format(part_name),
                translation = np.array(translation),
                orientation = euler_angles_to_quat(rotation),
                scale = scale
            )
        )
    )
    prim = get_prim_by_path(stage, prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    PhysxSchema.PhysxContactReportAPI.Apply(prim)

    mesh_path = "/World/parts/{}/{}/mesh".format(part_name, part_name)
    prim_mesh = get_prim_by_path(stage, mesh_path)

    colAPI = PhysxSchema.PhysxSDFMeshCollisionAPI(prim_mesh)
    colAPI.CreateSdfMarginAttr().Set(0.0)

    omni.kit.commands.execute(
                "SetRigidBodyCommand", path=prim_path, approximationShape="sdf", kinematic=False
            )

    colAPI.GetSdfResolutionAttr().Set(128)
    colAPI.GetSdfEnableRemeshingAttr().Set(True)

    if sdf:
        colAPI.GetSdfTriangleCountReductionFactorAttr().Set(0.5)

    geometry_prim = SingleGeometryPrim(prim_path, name="{}_geom".format(part_name))
    geometry_prim.apply_physics_material(PhysicsMaterial(PHYSICS_MATERIAL_PATH))

    estimated_mass = density * volume
    print("Volume:", volume, "estimated mass:", estimated_mass)
    MassAPI = UsdPhysics.MassAPI.Apply(prim)
    MassAPI.GetMassAttr().Set(estimated_mass)

#--------------------------------------------------------------------#
class ContactTracker:
    """Records what the craft touches while the function is being executed.

    Two things matter: which parts sit on the ground at the start (the craft's
    footprint) and whether anything else drops onto it later, which means the
    craft collapsed. Contacts between parts are collected as feedback for the
    next prompt.
    """

    def __init__(self):
        self.initial_ground_contact = []
        self.all_ground_contact = []
        self.all_parts_contact = []
        self.count_ground_contact = []
        self.feedback_msg = ""
        self.ground_c = False
        self.curr_ts = 0

    def callback(self, contact_headers, contact_data):
        add_ground = self.initial_ground_contact == []

        for header in contact_headers:
            act0_path = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            act1_path = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            part_1_ = act0_path.split("/")[-1]
            part_2_ = act1_path.split("/")[-1]
            touches_ground = GROUND_PLANE_PATH in (act0_path, act1_path)

            if not touches_ground:
                already_added = any(
                    (act0_path in pair) and (act1_path in pair) for pair in self.all_parts_contact
                )
                if not already_added:
                    self.all_parts_contact.append([act0_path, act1_path])
                    self.feedback_msg += (
                        f"Detected contact between the following parts: "
                        f"{part_1_} and {part_2_}. \n"
                    )
                continue

            if add_ground:
                self.initial_ground_contact.append([act0_path, act1_path])
                self.all_ground_contact.append([act0_path, act1_path])
                self.count_ground_contact.append(1)
                self.feedback_msg += (
                    f"The initial contacts with the ground are: {part_1_} and {part_2_}. \n"
                )
                continue

            self._track_later_ground_contact(act0_path, act1_path, part_1_, part_2_)

    def _track_later_ground_contact(self, act0_path, act1_path, part_1_, part_2_):
        for idx, pair in enumerate(self.all_ground_contact):
            if (act0_path in pair) and (act1_path in pair):
                self.count_ground_contact[idx] += 1
                # Wheels are meant to be on the ground; anything else is a fall.
                is_wheel = "wheel" in part_1_.lower() or "wheel" in part_2_.lower()
                if not is_wheel and self.count_ground_contact[idx] == GROUND_CONTACT_PATIENCE:
                    self.ground_c = True
                    self.feedback_msg += (
                        f"The following contact with the ground happened during the "
                        f"function execution: {part_1_} and {part_2_}. \n"
                    )
                    print(self.feedback_msg)
                    print(self.curr_ts)
                return

        self.all_ground_contact.append([act0_path, act1_path])
        self.count_ground_contact.append(1)

    def summary(self):
        """The contacts, phrased for the re-prompt feedback."""
        message = ""
        for contacts, prefix in (
            (self.initial_ground_contact, "The initial contacts with the ground are: "),
            (self.all_ground_contact,
             "During the function execution, the following contacts with the ground happened: "),
            (self.all_parts_contact,
             "During the function execution, the following parts had contact with each other: "),
        ):
            pairs = "".join(
                f"[{pair[0].split('/')[-1]}, {pair[1].split('/')[-1]}], " for pair in contacts
            )
            if pairs:
                message += prefix + pairs + "\n "
        return message


#--------------------------------------------------------------------#
def setup_world(folder_name, individual):
    """A fresh stage with a ground plane, and the folders the assets go through."""
    stage = omni.usd.get_context().get_stage()
    my_world = World(stage_units_in_meters=1.0)
    scene = my_world.scene
    scene.add_default_ground_plane()

    ground_plane = stage.GetPrimAtPath(GROUND_PLANE_PATH)
    ground_plane.GetAttribute('physics:approximation').Set("sdf")

    # folder path for 3d assets
    input_folder = os.path.join(folder_name, "options" if individual else "centered")
    output_folder = os.path.join(folder_name, "assets")

    # convert .obj, .stl or .fbx to .usd
    for f in glob.glob(output_folder + '/*'):
        os.remove(f)

    return stage, my_world, scene, input_folder, output_folder


def hit_orientation(parts_names, folder_name):
    """How to stand a hammer up: its longest part decides which way is 'along'."""
    max_dim = 0
    arg_max = 0
    for counter, part in enumerate(parts_names):
        p_mesh = trimesh.load_mesh(os.path.join(folder_name, "final", part + ".obj"))
        dims_curr = p_mesh.bounds[1] - p_mesh.bounds[0]
        if any(dims_curr > max_dim):
            max_dim = np.max(dims_curr)
            arg_max = counter

    max_mesh = trimesh.load_mesh(os.path.join(folder_name, "final", parts_names[arg_max] + ".obj"))
    dims = max_mesh.bounds[1] - max_mesh.bounds[0]
    longest_axis = np.argmax(dims)

    if longest_axis == 0:
        return np.array([np.pi, 0, 0]), None
    if longest_axis == 1:
        return np.array([0, 0, -np.pi / 2]), 1
    return np.array([0, -np.pi / 2, 0]), 2


def spawn_parts(parts_names, volumes, density_list, sdf_list, input_folder, output_folder,
                folder_name, function_name, case, move_up, scene, stage, added_parts, rotation):
    """Convert every part to USD and drop it into the stage at its authored pose."""
    for part, vol, density, sdf in zip(parts_names, volumes, density_list, sdf_list):
        convert_to_usd(os.path.join(input_folder, part + ".obj"),
                       os.path.join(output_folder, prim_name(part) + ".usd"))

        # spawn model in ISAAC
        p_mesh = trimesh.load_mesh(os.path.join(folder_name, "final", part + ".obj"))
        t = (p_mesh.bounds[1] + p_mesh.bounds[0]) / 2.0
        if function_name == "hit":
            # The hammer is re-stood on the axis hit_orientation picked, and
            # lifted clear of the nail.
            t_copy = abs(p_mesh.bounds[1] + p_mesh.bounds[0]) / 2.0
            if case == 1:
                t[0], t[1], t[2] = t_copy[1], t_copy[0], t_copy[2] + 18 / 100
            elif case == 2:
                t[0], t[1], t[2] = t_copy[2], t_copy[1], t_copy[0] + 18 / 100
            else:
                t[2] = t_copy[2] + 18 / 100
        else:
            t[2] = t[2] + move_up

        add_part(part, t, rotation, scene, stage, added_parts, vol, density, folder_name,
                 scale=np.array([1.0, 1.0, 1.0]), sdf=sdf)


def separated_from_craft(stage, parts_names, root_pos, initial_dists, thresholds, separated_list):
    """Whether any part has drifted away from the craft, and which ones.

    ``thresholds`` is either one distance for every part or one per part.
    """
    if not isinstance(thresholds, (list, tuple)):
        thresholds = [thresholds] * len(parts_names)

    separated = False
    for part, dist_init, threshold in zip(parts_names, initial_dists, thresholds):
        part_prim = stage.GetPrimAtPath(part_path(part))
        part_pos = np.array(part_prim.GetAttribute('xformOp:translate').Get())
        if abs(distance(part_pos, root_pos) - dist_init) > threshold:
            separated = True
            if prim_name(part) not in separated_list:
                separated_list.append(prim_name(part))
    return separated


#--------------------------------------------------------------------#
def create_fixed_joints(stage, connection_list, connection_type):
    """Weld every FIXED pair together, anchored at the pose they were spawned in.

    A joint's local frames default to the identity on both bodies, which asks
    PhysX to make the two origins coincide -- it warns about "disjointed body
    transforms" and snaps the craft into a heap on the ground. Authoring the
    frames from the parts' relative transform holds the craft in the shape the
    plan describes.
    """
    counter_joint = 0
    for part_pair, connection in zip(connection_list, connection_type):
        if connection != "FIXED":
            continue
        body_0_path = part_path(part_pair[0])
        body_1_path = part_path(part_pair[1])

        fixed_joint = UsdPhysics.FixedJoint.Define(
            stage, "{}/FixedJoint_{}".format(body_1_path, counter_joint))
        fixed_joint.CreateBody0Rel().SetTargets([body_0_path])
        fixed_joint.CreateBody1Rel().SetTargets([body_1_path])

        # Body 1's pose expressed in body 0's frame. USD matrices are
        # row-vector, so composing world-to-local is M1 * M0^-1.
        to_world_0 = UsdGeom.Xformable(stage.GetPrimAtPath(body_0_path)) \
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        to_world_1 = UsdGeom.Xformable(stage.GetPrimAtPath(body_1_path)) \
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        relative = to_world_1 * to_world_0.GetInverse()

        fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(relative.ExtractTranslation()))
        fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(relative.ExtractRotationQuat()))
        fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

        counter_joint += 1
    return counter_joint


#--------------------------------------------------------------------#
def physics_test(function_name, parts_names, volumes, move_up, folder_name, functional_list, root_part=None,
                 connection_list=None, connection_type=None, individual=False, rotation_list=None,
                 density_list=None, sdf_list=None, utils_root=""):
    """Simulate one craft performing ``function_name``.

    Returns ``(passed, fail_reason, feedback_msg)``; the two strings are what the
    re-prompt quotes back to the model.
    """
    stage, my_world, scene, input_folder, output_folder = setup_world(folder_name, individual)

    added_parts = []
    r = np.array([0, 0, 0])
    case = None
    if function_name == "hit":
        r, case = hit_orientation(parts_names, folder_name)

    spawn_parts(parts_names, volumes, density_list, sdf_list, input_folder, output_folder,
                folder_name, function_name, case, move_up, scene, stage, added_parts, r)

    # Adjust physics material
    material_prim = stage.GetPrimAtPath(PHYSICS_MATERIAL_PATH)
    material_prim.GetAttribute('physics:dynamicFriction').Set(0.1)
    material_prim.GetAttribute('physics:staticFriction').Set(0.2)
    material_prim.GetAttribute('physics:restitution').Set(0.3)

    # create Fixed joints for "fixed" connection
    if not individual:
        create_fixed_joints(stage, connection_list, connection_type)

    # Setup ForceAPI inside ISAAC that should be common for any function
    result = False

    driven = prim_name(root_part) if not individual else prim_name(parts_names[0])
    force_prim = stage.GetPrimAtPath("/World/parts/{}".format(driven))
    force_mesh = stage.GetPrimAtPath("/World/parts/{}/{}/mesh".format(driven, driven))

    initial_pos = np.array(force_prim.GetAttribute('xformOp:translate').Get())

    functional_prims = [prim_name(part) for part, functional in zip(parts_names, functional_list)
                        if functional]
    all_parts = [prim_name(part) for part in parts_names]

    # setup the force attribute to test the function
    forceApi = PhysxSchema.PhysxForceAPI.Apply(force_mesh)
    forceAttr = forceApi.GetForceAttr()
    forceApi.GetModeAttr().Set("force")
    forceApi.GetForceEnabledAttr().Set(True)
    forceApi.GetWorldFrameEnabledAttr().Set(True)

    # Distances from every part to the driven part, so a part coming loose during
    # the run can be told apart from the craft simply moving.
    all_initial_dists = []
    all_dist_thresholds = []
    total_mass = 0
    for part in all_parts:
        part_prim = stage.GetPrimAtPath("/World/parts/{}".format(part))
        total_mass += part_prim.GetAttribute('physics:mass').Get()

        part_pos = np.array(part_prim.GetAttribute('xformOp:translate').Get())
        dist_part = distance(part_pos, initial_pos)
        all_initial_dists.append(dist_part)
        all_dist_thresholds.append(1.5 * dist_part)

    estimated_force = total_mass * 9.89 * 0.10
    print("Total mass:", total_mass, "estimated force:", estimated_force)

    physics_scene = stage.GetPrimAtPath("/physicsScene")
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
    physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
    physx_scene.CreateTimeStepsPerSecondAttr().Set(500)

    attr = physics_scene.CreateAttribute("physxScene:enableContactReports", Sdf.ValueTypeNames.Bool)
    attr.Set(True)

    contacts = ContactTracker()
    # The subscription has to stay referenced for the whole run: dropping it
    # unsubscribes the callback and the contact record comes back empty.
    contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(
        contacts.callback)

    fail_reason = ""
    format_feed_msg = ""
    separated_list = []

    if function_name == "rolling":
        result, fail_reason, format_feed_msg = run_rolling(
            my_world, stage, contacts, forceAttr, estimated_force, force_prim, initial_pos,
            parts_names, functional_prims, rotation_list, all_initial_dists, all_dist_thresholds,
            separated_list)

    elif function_name == "support":
        result, fail_reason = run_support(
            my_world, stage, forceAttr, estimated_force, force_prim, initial_pos,
            parts_names, all_initial_dists, separated_list)

    elif function_name == "hit":
        result, fail_reason = run_hit(
            my_world, stage, scene, contacts, added_parts, force_prim, functional_prims, all_parts,
            parts_names, all_initial_dists, separated_list, utils_root)

    elif function_name == "stability":
        result, fail_reason = run_stability(my_world, contacts)

    my_world.clear_instance()
    scene.clear()
    prims_utils.delete_prim(PHYSICS_MATERIAL_PATH)

    return result, fail_reason, format_feed_msg + contacts.summary()


def run_rolling(my_world, stage, contacts, forceAttr, estimated_force, force_prim, initial_pos,
                parts_names, functional_prims, rotation_list, all_initial_dists,
                all_dist_thresholds, separated_list):
    """Push the craft forward: it must travel, on wheels that actually turn."""
    scale = 1e-3
    dist_min = 50.0 * scale
    dist_side = dist_min

    all_angVel = [stage.GetPrimAtPath("/World/parts/{}".format(prim))
                  .GetAttribute('physics:angularVelocity') for prim in functional_prims]

    # The axis each functional part is expected to spin about, in world frame.
    all_rot_axis = [np.matmul(rotation, np.array([0, 0, 1])) for rotation in rotation_list]
    sum_angVel = [np.array([0.0, 0.0, 0.0]) for _ in functional_prims]

    forceAttr.Set(Gf.Vec3f(estimated_force, 0.0, 0.0))
    count_angvel = 0
    separated_parts = False
    final_pos = initial_pos

    my_world.render()
    my_world.play()
    while my_world.is_playing():
        my_world.render()
        my_world.step()
        contacts.curr_ts += 1

        if contacts.curr_ts % 10 == 0:
            root_pos = force_prim.GetAttribute('xformOp:translate').Get()
            if separated_from_craft(stage, parts_names, root_pos, all_initial_dists,
                                    all_dist_thresholds, separated_list):
                separated_parts = True

        if my_world.current_time_step_index >= FINAL_STEP + 2:
            final_pos = force_prim.GetAttribute('xformOp:translate').Get()
            my_world.stop()
        elif contacts.curr_ts % 10 == 0:
            for j in range(len(sum_angVel)):
                sum_angVel[j] += np.array(all_angVel[j].Get())
                count_angvel += 1

    dist = abs(final_pos[0] - initial_pos[0])
    dist_others = np.sqrt((final_pos[1] - initial_pos[1]) ** 2 + (final_pos[2] - initial_pos[2]) ** 2)

    # At least one full turn, about the axis the wheel was authored to spin on.
    rot = True
    fail_func = []
    for ang, f_prim, rot_a in zip(np.array(sum_angVel) / count_angvel, functional_prims,
                                 all_rot_axis):
        measured_rot = abs(np.dot(ang, rot_a))
        if measured_rot < 2 * np.pi or np.isnan(measured_rot):
            fail_func.append(f_prim)
            rot = False

    result = (dist > dist_min) and rot and (not contacts.ground_c) and (not separated_parts)
    print(f"dist={dist} rot={rot} dist_others={dist_others} ground={contacts.ground_c} "
          f"separated={separated_parts} result={result}")

    fail_reason = ""
    if dist < dist_min:
        fail_reason += "The craft did not move forward. "
    if not rot:
        for f_prim in fail_func:
            fail_reason += f"{f_prim} failed to rotate. "
    if dist_others > dist_side:
        fail_reason += "The craft veered to the side. "
    if separated_parts:
        fail_reason += (f"The following parts got separated from the craft during the function "
                        f"execution: {separated_list}.")

    format_feed_msg = ""
    if (dist < dist_min) and (not rot):
        format_feed_msg = (f"The main axis of the cylindrical functional parts"
                           f"({functional_prims}) are {all_rot_axis}.")

    return result, fail_reason, format_feed_msg


def run_support(my_world, stage, forceAttr, estimated_force, force_prim, initial_pos,
                parts_names, all_initial_dists, separated_list):
    """Press down on the craft: it must not move and must not come apart."""
    forceAttr.Set(Gf.Vec3f(0.0, 0.0, -estimated_force))
    separated_parts = False
    # A run that never reaches the final step has not shown the craft is stable.
    travelled = float("inf")

    my_world.render()
    my_world.play()
    while my_world.is_playing():
        my_world.render()
        my_world.step()
        root_pos = force_prim.GetAttribute('xformOp:translate').Get()
        if separated_from_craft(stage, parts_names, root_pos, all_initial_dists, 0.005,
                                separated_list):
            separated_parts = True

        if my_world.current_time_step_index >= FINAL_STEP + 2:
            final_pos = force_prim.GetAttribute('xformOp:translate').Get()
            my_world.stop()
            travelled = distance(final_pos, initial_pos)

    print("Travelled:", travelled)
    fail_reason = ""
    result = travelled < 0.001
    if not result:
        fail_reason += ("The craft doesn't seem stable. It moved when applying a downward force "
                        "to the functional parts.")

    if separated_parts:
        result = False
        fail_reason += (f"The following parts got separated during the function execution: "
                        f"{separated_list}.")

    return result, fail_reason


def run_hit(my_world, stage, scene, contacts, added_parts, force_prim, functional_prims, all_parts,
            parts_names, all_initial_dists, separated_list, utils_root):
    """Drop the craft on a nail: the functional part must strike it home."""
    # Balance -> The "hit" part needs to be heavier
    # Hold the object by the center of the length
    func_prim = stage.GetPrimAtPath("/World/parts/{}".format(functional_prims[0]))
    initial_pos = func_prim.GetAttribute('xformOp:translate').Get()

    asset_convert(os.path.join(utils_root, "meshes"), os.path.join(utils_root, "assets"))
    vol_nail = 2 / 3 * np.pi * (0.01 ** 3) + np.pi * 0.005 * 0.005 * 0.02
    vol_board = 0.05 * 0.05 * 0.05 - np.pi * 0.005 * 0.005 * 0.02
    add_part("nail", np.array([initial_pos[0], initial_pos[1], 0.060]), np.array([0, 0, 0.0]),
             scene, stage, added_parts, vol_nail, 1030, utils_root,
             scale=np.array([0.01, 0.01, 0.01]))
    add_part("board", np.array([initial_pos[0], initial_pos[1], 0.025]), np.array([0, 0, 0.0]),
             scene, stage, added_parts, vol_board, 1030, utils_root,
             scale=np.array([0.015, 0.015, 0.01]))

    nail_prim = get_prim_by_path(stage, "/World/parts/nail")
    board_prim = get_prim_by_path(stage, "/World/parts/board")
    nail_prim.GetAttribute('physxRigidBody:disableGravity').Set(True)
    # lockPos profiles are a bit field over X, Y, Z; 7 locks all three.
    board_prim.GetAttribute('physxRigidBody:lockedPosAxis').Set(7)
    board_prim.GetAttribute('physxRigidBody:lockedRotAxis').Set(7)

    nail_pos = nail_prim.GetAttribute('xformOp:translate').Get()
    nail_target = np.array([nail_pos[0], nail_pos[1], 0.030])

    all_prims = [stage.GetPrimAtPath("/World/parts/{}".format(p)) for p in all_parts]
    check_contact = True
    separated_parts = False
    # A run that never reaches the final step has not driven the nail anywhere.
    travelled = float("inf")

    my_world.render()
    my_world.play()
    while my_world.is_playing():
        my_world.step()

        # Once the craft reaches the nail the swing is over: freeze the craft and
        # let the nail alone respond, so only the blow that landed is measured.
        if check_contact:
            for p_contact in contacts.all_parts_contact:
                names = [p_contact[0].split("/")[-1], p_contact[1].split("/")[-1]]
                if "nail" in names:
                    for prim in all_prims:
                        prim.GetAttribute('physxRigidBody:disableGravity').Set(True)
                        prim.GetAttribute('physxRigidBody:lockedPosAxis').Set(7)
                        prim.GetAttribute('physxRigidBody:lockedRotAxis').Set(7)
                    check_contact = False
                    nail_prim.GetAttribute('physxRigidBody:disableGravity').Set(False)
                    break

        root_pos = force_prim.GetAttribute('xformOp:translate').Get()
        if separated_from_craft(stage, parts_names, root_pos, all_initial_dists, 0.005,
                                separated_list):
            separated_parts = True

        if my_world.current_time_step_index >= FINAL_STEP + 2:
            final_pos = nail_prim.GetAttribute('xformOp:translate').Get()
            my_world.stop()
            travelled = distance(final_pos, nail_target)

    fail_reason = ""
    failed_contact = False
    for func_p in functional_prims:
        hit_the_nail = any(
            "nail" in [pair[0].split("/")[-1], pair[1].split("/")[-1]]
            and func_p in [pair[0].split("/")[-1], pair[1].split("/")[-1]]
            for pair in contacts.all_parts_contact
        )
        if not hit_the_nail:
            failed_contact = True
            fail_reason += f"The following part did not hit the nail: {func_p}."
            break

    result = travelled < 0.005 and not failed_contact
    if not failed_contact and not result:
        fail_reason += "The craft didn't manage to hit the peg into the hole."

    if separated_parts:
        fail_reason += (f"The following parts got separated during the function execution: "
                        f"{separated_list}.")

    print("Nail travelled:", travelled)
    return result, fail_reason


def run_stability(my_world, contacts):
    """Leave the craft standing: nothing new may touch the ground."""
    my_world.render()
    my_world.play()
    while my_world.is_playing():
        my_world.step()
        if my_world.current_time_step_index >= STABILITY_FINAL_STEP + 2:
            my_world.stop()

    if not contacts.ground_c:
        return True, ""
    return False, "The craft is not stable."


#--------------------------------------------------------------------#
def load_pose_records(error_dir):
    """The connection/functional/root/rotation tables written by the mesh stage."""
    return {
        "connections": pd.read_json(os.path.join(error_dir, "connections.jsonl"), lines=True),
        "functional": pd.read_json(os.path.join(error_dir, "functional.jsonl"), lines=True),
        "root": pd.read_json(os.path.join(error_dir, "root.jsonl"), lines=True),
        "rotation": pd.read_json(os.path.join(error_dir, "rotation.jsonl"), lines=True),
        "position": pd.read_json(os.path.join(error_dir, "position.jsonl"), lines=True),
    }


def lookup(frame, custom_id, *columns):
    """The named columns of the row for ``custom_id``, or ``None`` if absent."""
    for i in range(len(frame["custom_id"])):
        if frame["custom_id"][i] == custom_id:
            return tuple(frame[column][i] for column in columns)
    return None


def stage_meshes(meshes_folder, file, part_names, folder_name, density, use_sdf):
    """Copy each part's mesh into the run folder, centred and un-centred.

    Returns the volumes, densities, per-part SDF flags and how far the craft has
    to be lifted to sit on the ground -- or ``None`` if a mesh has no usable
    volume, which the boolean stage should already have caught.
    """
    volumes = []
    densities = []
    sdf_list = []
    min_z = 100

    for part in part_names:
        # Modified/rebuilt variants take precedence over the original part mesh.
        mesh = None
        for suffix in ("_mod", "_new", ""):
            candidate = os.path.join(meshes_folder, file, part + suffix + ".obj")
            if os.path.isfile(candidate):
                mesh = trimesh.load_mesh(candidate)
                break
        if mesh is None:
            print(f"No mesh found for {part} in {file}")
            return None

        mesh.metadata.pop("name", None)
        mesh.export(os.path.join(folder_name, "final", part + ".obj"))

        densities.append(density)
        try:
            volumes.append(mesh.volume)
        except Exception as exc:
            print(f"Could not compute the volume of {part}: {exc}")
            return None

        min_z = min(min_z, mesh.bounds[0][2])

        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        mesh.apply_translation(-center)
        mesh.export(os.path.join(folder_name, "centered", part + ".obj"))

        sdf_list.append(use_sdf)

    return volumes, densities, sdf_list, min_z


def main():
    args = ARGS
    results_root = args.results_root
    obj_type = args.obj_type

    meshes_folder = f"{results_root}/{args.in_dir}/meshes_rescaled/{obj_type}"
    in_error_dir = f"{results_root}/{args.in_dir}/error/{obj_type}"
    out_error_dir = f"{results_root}/{args.out_dir}/error/{obj_type}"

    # Plans that already failed earlier in the pipeline are not worth simulating.
    with open(os.path.join(in_error_dir, "failures.json"), mode="r") as f:
        failure_json = json.load(f)
    with open(os.path.join(in_error_dir, "float.json"), mode="r") as f:
        float_json = json.load(f)

    error_json = pd.read_json(os.path.join(in_error_dir, "error.jsonl"), lines=True)
    error_keys = list(error_json["custom_id"]) if "custom_id" in error_json else []

    records = load_pose_records(in_error_dir)

    all_files = sorted(os.listdir(meshes_folder))
    all_files = all_files[args.start_index:args.start_index + args.max_files]

    success_counts = {name: 0 for name in list(CLASS_PREFIXES) + ["truck"]}
    success_list = []
    failure_dict = {}
    t_counter = 0

    for file in all_files:
        # ignore if failure before
        if file in failure_json.keys() or file in float_json.keys() or file in error_keys:
            continue

        print(file)

        folder_name = f"{results_root}/{args.out_dir}/physics/{obj_type}/{file}"
        if os.path.isdir(folder_name):
            continue
        for sub in ("final", "centered", "assets"):
            os.makedirs(os.path.join(folder_name, sub), exist_ok=True)

        obj_class = classify(file)
        function_name = CLASS_FUNCTION[obj_class]

        part_names = [x.split(".")[0] for x in os.listdir(os.path.join(meshes_folder, file))]

        connections = lookup(records["connections"], file, "connection_pairs", "connection_type")
        functional = lookup(records["functional"], file, "parts")
        root = lookup(records["root"], file, "part")
        rotations = lookup(records["rotation"], file, "parts", "values")
        if any(record is None for record in (connections, functional, root, rotations)):
            print(f"Missing pose records for {file}, skipping")
            continue

        connection_pairs, connection_type = connections
        functional_list = [x in functional[0] for x in part_names]
        root_part = root[0]
        rotation_list_parts, rotation_list_rot = rotations
        print(part_names)
        print(functional_list)

        rotation_list = [rotation_list_rot[rotation_list_parts.index(name)]
                         for name in part_names if name in rotation_list_parts]

        staged = stage_meshes(meshes_folder, file, part_names, folder_name,
                              args.density, args.use_sdf)
        if staged is None:
            result, fail_reason, feedback_msg = False, "A part has no usable volume.", ""
        else:
            volumes, densities, sdf_list, min_z = staged

            if function_name == "rolling":
                rot_list = [R.from_euler("xyz", rot or [0, 0, 0], degrees=True).as_matrix()
                            for func, rot in zip(functional_list, rotation_list) if func]
            else:
                rot_list = None

            result, fail_reason, feedback_msg = physics_test(
                function_name=function_name, parts_names=part_names, volumes=volumes,
                move_up=-min_z, folder_name=folder_name, functional_list=functional_list,
                root_part=root_part, connection_list=connection_pairs,
                connection_type=connection_type, rotation_list=rot_list,
                density_list=densities, sdf_list=sdf_list, utils_root=args.utils_root)
            print(result)

        if result:
            success_counts[obj_class] += 1
            success_list.append(file)
        else:
            print(fail_reason)
            failure_dict[file] = [{"failure_reason": fail_reason},
                                  {"additional_info": feedback_msg}]

        t_counter += 1
        for name, count in success_counts.items():
            print(f"The success cases for {name.capitalize()} are: {count}")
        print("The total cases so far is: ", t_counter)
        print(success_list)

    # Creating the file
    os.makedirs(out_error_dir, exist_ok=True)
    out_file = os.path.join(out_error_dir, "physics.jsonl")
    with open(out_file, 'w') as o_file:
        for key in failure_dict.keys():
            o_file.write(json.dumps({key: failure_dict[key]}) + '\n')
    print(f"Wrote {len(failure_dict)} failures -> {out_file}")


if __name__ == "__main__":
    main()
