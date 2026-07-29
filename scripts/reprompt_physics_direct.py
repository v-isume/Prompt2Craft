import pandas as pd
from pydantic import BaseModel  # for structured output from python class
import json
import os
import numpy as np

# openai API
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

from openai import OpenAI
client = OpenAI()

available_dims = {
"box_A" : [100,  10,  10, "cuboid"],
"box_B" : [75,   10,  10, "cuboid"],
"box_C" : [50,   10,  10, "cuboid"],
"box_D" : [95,   95,  10, "cuboid"],
"box_E" : [70,   70,  10, "cuboid"],
"box_F" : [40,   40,  10, "cuboid"],
"box_G" : [95,   95,  95, "cuboid"],
"box_H" : [75,   70,  75, "cuboid"],
"box_I" : [45,   45,  45, "cuboid"],
"box_J" : [100,  70,  10, "cuboid"],
"box_K" : [100,  45,  10, "cuboid"],
"box_L" : [75,   45,  10, "cuboid"],
"box_M" : [100,  70,  75, "cuboid"],
"box_N" : [150,  70, 100, "cuboid"],
"box_O" : [150,  95,  40, "cuboid"],
"box_P" : [100,  70,  40, "cuboid"],
"box_Q" : [150,  70,  40, "cuboid"],
"box_R" : [150,  45,  40, "cuboid"],
"box_S" : [150,  45,  10, "cuboid"],
"box_T" : [35,   10,  10, "cuboid"],
"box_U" : [35,   35,  10, "cuboid"],
"box_V" : [25,   25,  10, "cuboid"],
"cyl_A" : [50,   50,  10, "cylinder"],
"cyl_B" : [25,   25,  10, "cylinder"],
"cyl_C" : [15,   15,  10, "cylinder"],
"cyl_D" : [50,   50,  40, "cylinder"],
"cyl_E" : [25,   25,  15, "cylinder"],
"cyl_F" : [20,   20, 100, "cylinder"],
"cyl_G" : [10,   10, 100, "cylinder"],
"cyl_H" : [10,   10,  50, "cylinder"],
"cyl_I" : [20,   20,  50, "cylinder"],
"cyl_J" : [20,   20,  75, "cylinder"],
"cyl_K" : [10,   10,  75, "cylinder"],
"cyl_M" : [20,   20, 150, "cylinder"],
"cyl_N" : [30,   30, 200, "cylinder"],
"cyl_O" : [30,   30, 250, "cylinder"],
"cyl_P" : [20,   20, 175, "cylinder"],
"cyl_Q" : [30,   30, 125, "cylinder"],
"cyl_R" : [30,   30, 225, "cylinder"],
"cyl_S" : [15,   15,  25, "cylinder"],
"cyl_T" : [15,   15,  50, "cylinder"]
}

available_objs_str = ""
for key in available_dims.keys():
    available_objs_str += f"{key} is a {available_dims[key][3]} with the following dimensions: [{available_dims[key][0]} mm, {available_dims[key][1]} mm, {available_dims[key][2]} mm] \n"

class PlanFormat(BaseModel):
    order_part_names: list[str]
    corresponding_obj_list: list[str]
    position_list: list[list[float]]
    rotation_list: list[list[float]]
    connection_pairs: list[list[str]]
    connection_type: list[str]
    boolean_pairs: list[list[str]]
    functional_parts: list[bool]

model = "o4-mini"

obj_type = "truck"

prompt_json_path    = f"./json_final/o4-mini_batch_{obj_type}.jsonl"
answer_json_path    = f"./json_final/o4-mini_batch_{obj_type}_answer.jsonl"
physics_json_path = f"/media/vitor/SSD-1/CraftGPT_results/new/initial/error/{obj_type}/physics.jsonl"

re_prompt_json_path    = f"./json_final/feedback/o4-mini_batch_{obj_type}_reprompt_collision.jsonl"
re_answer_json_path    = f"./json_final/feedback/o4-mini_batch_{obj_type}_reprompt_collision_answer.jsonl"
re_physics_json_path = f"/media/vitor/SSD-1/CraftGPT_results/new/collision/error/{obj_type}/physics.jsonl"

all_tasks = []

prompt_json = pd.read_json(prompt_json_path, lines=True)
answer_json = pd.read_json(answer_json_path, lines=True)

re_prompt_json = pd.read_json(prompt_json_path, lines=True)
re_answer_json = pd.read_json(answer_json_path, lines=True)

physics_json_list = []
re_physics_json_list = []

with open(physics_json_path, mode="r") as f:
    lines = f.readlines()
    for line in lines:
        curr_dict = json.loads(line)
        physics_json_list.append(curr_dict)

with open(re_physics_json_path, mode="r") as f:
    lines = f.readlines()
    for line in lines:
        curr_dict = json.loads(line)
        re_physics_json_list.append(curr_dict)

all_images = ["0380.jpg", "0441.jpg"]
found_list = [False, False]
counter = 0

for img in all_images:
    
    messages = []

    for j in range(len(prompt_json)):
        if prompt_json["custom_id"][j] == img:
           messages = prompt_json["body"][j]["messages"]

    for k in range(len(answer_json)):
        if answer_json["custom_id"][k] == img:
            reply = answer_json["response"][k]["body"]["choices"][0]["message"]
            messages.append(reply)
    
    for i in range(len(physics_json_list)):
        if list(physics_json_list[i].keys())[0] == img:
            idx = i
            found_list[counter] = True
            break
    
    if not found_list[counter]:
        counter += 1
        continue
        
    counter +=1

    fail_reason = physics_json_list[idx][img][0]["failure_reason"]
    add_reason  = physics_json_list[idx][img][1]["additional_info"]

    united_reason = fail_reason + add_reason

    user_prompt = f"""This craft plan is not valid because of the following: {united_reason}.
    Create a new craft assembly plan, from scratch, while avoiding the errors from the previous plan.
    """

    prompt = {"role": "user",
            "content": [{"type": "text",
                        "text": user_prompt}]}
    
    messages.append(prompt)

    
    request_data = {
                "model": "o4-mini",
                "reasoning_effort": "high",
                "messages": messages,
                "response_format": {"type": "json_object"}
    }

    response = client.chat.completions.create(**request_data)

    log_entry = {
        "custom_id": img,
        "request": request_data,
        "response": response.model_dump()  # important!
    }

    with open(f"./json_new/physics/{obj_type}_new.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

counter = 0
for img in all_images:
    messages = []

    if found_list[counter]:
        counter += 1
        continue

    for j in range(len(re_prompt_json)):
        if re_prompt_json["custom_id"][j] == img:
           messages = re_prompt_json["body"][j]["messages"]

    for k in range(len(re_answer_json)):
        if re_answer_json["custom_id"][k] == img:
            reply = re_answer_json["response"][k]["body"]["choices"][0]["message"]
            messages.append(reply)
    
    for i in range(len(re_physics_json_list)):
        if list(re_physics_json_list[i].keys())[0] == img:
            idx = i
            found = True
            break
    
    if not found:
        counter += 1
        continue
    
    counter += 1

    fail_reason = re_physics_json_list[idx][img][0]["failure_reason"]
    add_reason  = re_physics_json_list[idx][img][1]["additional_info"]

    united_reason = fail_reason + add_reason

    user_prompt = f"""This craft plan is not valid because of the following: {united_reason}.
    Create a new craft assembly plan, from scratch, while avoiding the errors from the previous plan.
    """

    prompt = {"role": "user",
            "content": [{"type": "text",
                        "text": user_prompt}]}
    
    messages.append(prompt)

    
    request_data = {
                "model": "o4-mini",
                "reasoning_effort": "high",
                "messages": messages,
                "response_format": {"type": "json_object"}
    }

    response = client.chat.completions.create(**request_data)

    log_entry = {
        "custom_id": img,
        "request": request_data,
        "response": response.model_dump()  # important!
    }

    with open(f"./json_new/physics/{obj_type}_new.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")