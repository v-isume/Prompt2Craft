"""Prompt for assembly plans one image at a time, instead of via the batch API.

Same job as ``batch_prompt.py``, but each image is sent as a live request and its
plan written straight to disk. Use it for a small folder or a quick iteration on
the prompt, where waiting on a batch job is not worth it.

The prompt itself is a different dialect from ``batch_prompt.py``: modifications
name the face they are cut from (``FRONT_MOD`` ...) and insertions carry an
explicit ``insertion_depth``. This is the vocabulary ``mesh_get_single.py``
reads, so the two scripts must stay in step -- ``batch_prompt.py`` and
``craft_assembly.py`` speak the older ``FRONT_BACK_FULL`` form and are not
interchangeable with them.

The API key is read from the ``OPENAI_API_KEY`` environment variable.
Runtime settings default to ``config/prompt_single_config.json`` and may be
overridden on the command line.

Usage:
    python prompt_single_robot.py
    python prompt_single_robot.py --obj-type robot --img-folder DIR
"""
import argparse
import json
import os

from openai import OpenAI

from batch_prompt import encode_image, get_available_objs_str, get_obj_parameters, get_template
from config_loader import load_available_dims, load_config


def generate_prompt(obj_type, available_dims, object_parameters, template_dir="."):
    """The full user prompt for one object type, in the ``_MOD`` dialect."""
    func_name, minimum_str, constraint_str = get_obj_parameters(obj_type, object_parameters)
    available_objs_str = get_available_objs_str(available_dims)
    template_json = get_template(obj_type, template_dir)

    template_description = f"This is one template for a {obj_type} that needs to perform {func_name} composed of {minimum_str}:"

    user_prompt = f"""
You are an expert in generating assembly plans for building crafts using available parts.
You will be given an image of a craft and a target function that the craft should perform.
Your task is to generate a plan to build a valid craft that is similar to the object in the image and needs to perform a target function.

{template_description}

{template_json}

"""

    user_prompt += """Each part is composed of:
modifications: Optional list. name: unique name for the modification, type: {"HOLE"}, align_x: Relative position in X-axis {FRONT, CENTER or BACK} or if the modification will be applied along this axis, specify the face where the modification should be applied {"FRONT_MOD", "BACK_MOD"} or MODIFICATION_NAME (only valid if referring to other modifications in the same part, it will use the same X-value of this modification), align_y: Relative position in Y-axis {RIGHT, CENTER or LEFT} or or if the modification will be applied along this axis, specify the face where the modification should be applied {"RIGHT_MOD", "LEFT_MOD"} or MODIFICATION_NAME (only valid if referring to other modifications in the same part, it will use the same Y-value of this modification), align_z: Relative position in Z-axis {HIGH, CENTER or LOW} or or if the modification will be applied along this axis, specify the face where the modification should be applied {"TOP_MOD", "BOTTOM_MOD"} or MODIFICATION_NAME (only valid if referring other modifications in the same part, it will use the same Z-value of this modification).
connections: Optional list. to_part: name of the part it is connected to, contact_type: {"SURFACE", "INSERTED"}, to_face: if contact_type = SURFACE -> face of the bounding box {TOP, BOTTOM, RIGHT, LEFT, FRONT, BACK} of the current part that will be connected to the other part specified in "to_part", i.e. if the value of "to_face" is "TOP" would mean that the "TOP" face of the current part will connect to the "BOTTOM" face of the "to_part", align_x: Relative alignment in X-axis {FRONT, CENTER or BACK} or CONTACT (if to_face is FRONT or BACK), align_y: Relative alignment in Y-axis {RIGHT, CENTER or LEFT} or CONTACT (if to_face is RIGHT or LEFT), align_z: Relative alignment in Z-axis {HIGH, CENTER or LOW} or CONTACT (if to_face is TOP or BOTTOM), type: type of the connection {FIXED, NOT_FIXED}
connections: Optional list. to_part: name of the part it is connected to, contact_type: {"SURFACE", "INSERTED"}, to_modification: if contact_type = INSERTED -> the name of the modification where the part will be inserted, type: type of the connection {FIXED, NOT_FIXED}, insertion_depth = one of {"ACROSS", "QUARTER", "HALF", "FULL"}; "ACROSS" means that the current part is expected to have the same center of the specified "to_part" across the modification axis. "ACROSS" cannot be used for multiple "INSERTED" connections for the same part; "QUARTER" means that the current part will only be inserted until a quarter of the length of the current part in the modification direction will be inserted;"HALF" means that the current part will only be inserted until half of the length of the current part in the modification direction will be inserted; "FULL" the part will be entirely inserted if the length permits it.
exec_function: Required. {TRUE, FALSE}, depending if this part should be checked for the specified function or not.
"""

    user_prompt += f"""Now, consider the following available objects types: {available_objs_str}.
First, analyze the image and the object in the image and the observed parts.
Prioritize similarity with the input image.
The attribute "AVAILABLE_OBJ" determines the chosen available object to represent the part.
In the case of a cuboid, "ORIENTATION" is a list [dim_x, dim_y, dim_z] determining the order of the dimensions of the part in the final pose. The rotations can only be orthogonal, meaning that you can change the order of the dimensions to reflect the rotation, but cannot change the dimension values from the selected available object, for example, if obj_J is selected: [100, 70, 10], the possible final orientations are [100, 70, 10] or [100, 10, 70] or [10, 100, 70] or [10, 70, 100] or [70, 10, 100] or [70, 100, 10].
In the case of a cylinder, "ORIENTATION" is a string (FRONT_BACK, LEFT_RIGHT or TOP_BOTTOM), indicating the direction of the principal axis of the cylinder.
If the part is INSERTED to a MODIFICATION, it should match the orientation of the MODIFICATION.
The minimum required for the {obj_type} is {minimum_str}, but you can add more parts if necessary.
These are the constraints for this specific object type: {constraint_str}

Adapt the previous template to a plan based on the input image, making sure to modify it by modifying the "AVAILABLE_OBJ" correspondence and adding parts to build a craft of that is similar to the instance in the provided image that needs to perform the function of {func_name}.
If the image deviates from the template significantly, think how the plan could be adapted to address those differences. Try to make the craft as close to the input image as possible, in terms of number of parts and proportions of the sizes of the parts.
You may also propose different mechanisms if necessary.
You may modify the parts names or include more parts that were not originally listed in the provided template.
The orientation is X+ = front, X- = back, Y+ = right, Y- = right, Z+ = up, Z- = down.
The craft should be oriented so it is along the X+ = front direction.

Focus on the appearance similarity with the instance in the image. Add more parts if necessary.

The connection between two parts should only be specified ONCE inside either of the parts "connection" attribute.
Two parts should not be connected to the same part in the same relative position and alignment.

Only output the json formatted plan. Do not output anything else.
"""

    return user_prompt


def request_plan(client, model, reasoning_effort, user_prompt, image_path):
    """Ask the model for the plan of one image and return it as a dict."""
    input_base64 = encode_image(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{input_base64}", "detail": "high"}},
        ],
    }]

    response = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def parse_args(defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj-type", default=defaults["obj_type"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--reasoning-effort", default=defaults["reasoning_effort"])
    parser.add_argument("--img-folder", default=None)
    parser.add_argument("--template-dir", default=defaults.get("template_dir", "."))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    defaults = load_config("prompt_single_config.json")
    args = parse_args(defaults)

    obj_type = args.obj_type
    img_folder = args.img_folder or defaults["img_folder"].format(obj_type=obj_type)
    output_dir = args.output_dir or defaults["output_dir"].format(obj_type=obj_type)

    available_dims = load_available_dims()
    object_parameters = load_config("object_parameters.json")
    user_prompt = generate_prompt(obj_type, available_dims, object_parameters, args.template_dir)

    client = OpenAI()
    os.makedirs(output_dir, exist_ok=True)

    for img in sorted(os.listdir(img_folder)):
        print(f"------ {img}")
        json_data = request_plan(
            client, args.model, args.reasoning_effort, user_prompt,
            os.path.join(img_folder, img),
        )

        out_path = os.path.join(output_dir, f"{img}_answer.json")
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(json_data, file, ensure_ascii=False, indent=2)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
