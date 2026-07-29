import pandas as pd
from pydantic import BaseModel  # for structured output from python class
import json
import os

# openai API
import base64

from config_loader import load_available_dims, load_object_parameters, load_config

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Available objects string to be included in the prompt
def get_available_objs_str(available_dims):
    available_objs_str = ""
    for key in available_dims.keys():
        available_objs_str += f"{key} is a {available_dims[key][3]} with the following dimensions: [{available_dims[key][0]} mm, {available_dims[key][1]} mm, {available_dims[key][2]} mm] \n"
    return available_objs_str

def get_obj_parameters(obj_type, object_parameters):
    if obj_type not in object_parameters:
        raise ValueError("Invalid object type")
    params = object_parameters[obj_type]
    return params["func_name"], params["minimum"], params["constraint"]

def get_template(obj_type, template_dir="."):
    template_file = os.path.join(template_dir, f"{obj_type}_template.json")
    with open(template_file, mode="r") as f:
        template_json = f.read()
    return template_json

def generate_prompt(obj_type, available_dims, object_parameters, template_dir="."):
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
    modifications: Optional list. name: unique name for the modification, type: {"HOLE"}, align_x: Relative position in X-axis {FRONT, CENTER or BACK} or if the modification will be applied along this axis, specify the direction and length ("FRONT_BACK_FULL", "FRONT_BACK_HALF", "BACK_FRONT_HALF") or MODIFICATION_NAME (only valid if referring to other modifications in the same part, it will use the same X-value of this modification), align_y: Relative position in Y-axis {RIGHT, CENTER or LEFT} or or if the modification will be applied along this axis, specify the direction and length ("RIGHT_LEFT_FULL", "RIGHT_LEFT_HALF", "LEFT_RIGHT_HALF") or MODIFICATION_NAME (only valid if referring to other modifications in the same part, it will use the same Y-value of this modification), align_z: Relative position in Z-axis {HIGH, CENTER or LOW} or or if the modification will be applied along this axis, specify the direction and length ("TOP_BOTTOM_FULL", "TOP_BOTTOM_HALF", "BOTTOM_TOP_HALF") or MODIFICATION_NAME (only valid if referring other modifications in the same part, it will use the same Z-value of this modification).
    connections: Optional list. to_part: name of the part it is connected to, contact_type: {"SURFACE", "INSERTED"}, to_face: if contact_type = SURFACE -> face of the bounding box {TOP, BOTTOM, RIGHT, LEFT, FRONT, BACK} of the current part that will be connected to the other part specified in "to_part", i.e. , align_x: Relative alignment in X-axis {FRONT, CENTER or BACK} or CONTACT (if to_face is FRONT or BACK), align_y: Relative alignment in Y-axis {RIGHT, CENTER or LEFT} or CONTACT (if to_face is RIGHT or LEFT), align_z: Relative alignment in Z-axis {HIGH, CENTER or LOW} or CONTACT (if to_face is TOP or BOTTOM), type: type of the connection {FIXED, NOT_FIXED}
    connections: Optional list. to_part: name of the part it is connected to, contact_type: {"SURFACE", "INSERTED"}, to_modification: if contact_type = INSERTED -> the name of the modification where the part will be inserted, type: type of the connection {FIXED, NOT_FIXED}. In case one part is inserted through multiple modifications, it will be placed such that the center of the part is in the middle point of the modifications in parts that are connected to other parts with "SURFACE" connections. If it is only one modification, it will be centered if the alignment ends in "_FULL", or it will be inserted as deep as possible, in case it ends with "_HALF".
    exec_function: Required. {TRUE, FALSE}, depending if this part should be checked for the specified function or not.

    The only possible modification is "HOLE", where a "HOLE" is drilled across the specified axis. In this case, specify the drilling direction by specifying either align_x in {"FRONT_BACK_FULL", "FRONT_BACK_HALF", "BACK_FRONT_HALF"} OR align_y in {"RIGHT_LEFT_FULL", "RIGHT_LEFT_HALF", "LEFT_RIGHT_HALF"} OR align_z in {"TOP_BOTTOM_FULL", "TOP_BOTTOM_HALF", "BOTTOM_TOP_HALF"}. The drilling direction has to be unique, meaning you cannot specify it for multiple directions.
    For the modifications alignment names, the first two words indicate where to start and the direction of the drill, while the last word if the length of the hole. For example, "FRONT_BACK_FULL" means it will start at the front (X+) and go towards the back (X-) and the hole will go across the entire object. In the case of "BACK_FRONT_HALF", the drilling will start from the back (X-) towards the front (X+) and will only drill until it reaches half of the dimension of the object in that direction.
    The type of connection can either be "FIXED" or "NOT FIXED".
    If there are multiple modifications or connections to the same face and same alignment in all three axis, the parts will be distributed to avoid collision if possible.
    For example, if an additional HOLE was added to a SUPPORT_FRONT part:

    "SUPPORT_FRONT": {
    modification: {
    name: "HOLE_1",
    type: "HOLE",
    align_x: "CENTER",
    align_y: "RIGHT_LEFT_FULL",
    align_z: "CENTER"},
    modification: {
    name: "HOLE_3",
    type: "HOLE",
    align_x: "CENTER",
    align_y: "RIGHT_LEFT_FULL",
    align_z: "CENTER"},
    exec_function: {
    value: FALSE}
    },

    HOLE_3 would be re-positioned to just in front of HOLE_1, since the proposed positioning is already occupied.
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

    The connection between two parts should only be specified ONCE inside either of the parts "connection" attribute.

    Only output the json formatted plan. Do not output anything else.
    """

    return user_prompt


if __name__ == "__main__":

    ###########################################
    # AVAILABLE OBJECTS
    ###########################################
    # The catalog is stored in config/available_dims.json.
    # The dimensions are in mm, and the order is [dim_x, dim_y, dim_z, type], where the type is either "cuboid" or "cylinder".
    # For the cylinders, dim_x and dim_y represent the diameter of the cylinder, while dim_z represents the height of the cylinder.
    # For cuboids, dim_x represents the dimension along the X-axis, dim_y along the Y-axis and dim_z along the Z-axis.
    available_dims = load_available_dims()
    object_parameters = load_object_parameters()

    ###########################################
    # USER PARAMETERS (config/batch_prompt_config.json)
    ###########################################
    config = load_config("batch_prompt_config.json")
    obj_type = config["obj_type"]              # options: "truck", "scooter", "skateboard", "chair", "table", "hammer", "bookshelf"
    model = config["model"]
    reasoning_effort = config["reasoning_effort"]  # options: "low", "medium", "high"
    template_dir = config.get("template_dir", ".")
    img_folder = config["img_folder"].format(obj_type=obj_type)  # folder with the input images

    ###########################################
    # PROMPT GENERATION
    ###########################################
    user_prompt = generate_prompt(obj_type, available_dims, object_parameters, template_dir)

    all_tasks = []
    all_images = os.listdir(img_folder)

    for img in all_images:
        input_base64 = encode_image(f"{img_folder}/{img}")

        prompt = {"role": "user",
                "content": [{"type": "text",
                            "text": user_prompt}, 
                            {"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{input_base64}",
                                                    "detail": "high"}}]}

        messages = []
        messages.append(prompt)

        task = {
        "custom_id": img,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "messages": messages,
            "response_format": {"type": "json_object"}
            }
        }

        all_tasks.append(task)

    # Creating the file
    file_name = config["output_jsonl"].format(model=model, obj_type=obj_type)

    with open(file_name, 'w') as file:
        for obj in all_tasks:
            file.write(json.dumps(obj) + '\n')
