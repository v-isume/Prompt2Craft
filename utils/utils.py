import os
import base64                   # for encoding images for multi-modal prompt
from pydantic import BaseModel  # for structured output from python class

# GPT method of encoding images in multi-modal prompt
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# make output dirs  
def make_out_dirs(out_dir):
  os.makedirs(out_dir, exist_ok=False)
  os.makedirs(os.path.join(out_dir, "assets"), exist_ok=True)
  os.makedirs(os.path.join(out_dir, "options"), exist_ok=True)
  os.makedirs(os.path.join(out_dir, "position"), exist_ok=True)
  os.makedirs(os.path.join(out_dir, "final"), exist_ok=True)

# update the token count
def update_token_usage(response, total_count_list, model_name, total_cost_value):
  in_tokens    = response.usage.prompt_tokens
  out_tokens   = response.usage.completion_tokens
  total_tokens = response.usage.total_tokens

  if model_name == "gpt-4o":
      total_cost = 0.00250 * in_tokens/1000 + 0.01000 * out_tokens/1000
  elif model_name == "o3-mini" or model_name == "o4-mini":
      total_cost = 0.00110 * in_tokens/1000 + 0.00440 * out_tokens/1000
  elif model_name == "gpt-4.1":
      total_cost = 0.00200 * in_tokens/1000 + 0.00800 * out_tokens/1000

  total_count_list[0] += in_tokens
  total_count_list[1] += out_tokens
  total_count_list[2] += total_tokens

  print("***" + "-" * 80 + "***")
  print(response.choices[0].message.content)
  print("Prompt tokens:", in_tokens)
  print("Completion tokens:", out_tokens)
  print("Total tokens:", total_tokens)
  print("Total cost (approximated):", "$" + str(total_cost))
  

  total_cost_value += total_cost
  print("Accumulated total cost (approximated):", "$" + str(total_cost_value))
  print("***" + "-" * 80 + "***")

  return total_count_list, total_cost_value


# classes for structured output
class PartNames(BaseModel):
  obj_class_name: str
  list_part_class: list[str]
  number_instances: list[int]
  reasoning: str

class PartNamesFuction(BaseModel):
  available_correspondence: list[str]
  list_part_class: list[str]
  number_instances: list[int]
  functional_part_class: list[str]

class ConnectionPairs(BaseModel):
  connection_pairs: list[list[str]]
  connection_type: list[str]

class BooleanOperations(BaseModel):
  boolean_pairs: list[list[str]]

class RotationList(BaseModel):
  euler_list: list[list[float]]

class Translation(BaseModel):
  list_position: list[list[float]]