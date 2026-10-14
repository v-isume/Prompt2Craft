from pydantic import BaseModel  # for structured output from python class

# Structure for GPT to predict target object class, a list of part classes that make up the object and number of instances
class PartNames(BaseModel):
   obj_class_name: str
   list_part_class: list[str]
   number_instances: list[int]
   reasoning: str

# Structure for GPT to predict target object class, a list of part classes that make up the object and number of instances
class PartNamesFuction(BaseModel):
   list_part_class: list[str]
   number_instances: list[int]
   functional_part_class: list[str]

class AvailableObjCorrespondence(BaseModel):
    available_correspondence: list[str]
    connection_pairs: list[list[str]]

class PairsBooleanOperation(BaseModel):
    pairs_modification: list[list[str]]

class RotationList(BaseModel):
    orientation_final: list[list[float]]

class RotationChoices(BaseModel):
    rotation_choices: list[int]

class BooleanOperations(BaseModel):
    boolean_pairs: list[list[str]]

class TranslationReason(BaseModel):
    reasoning: str
    values: list[float]

class ReadjustReason(BaseModel):
    reasoning: str
    part_name: str
    values: list[float]

class FinalCheck(BaseModel):
    reasoning: str
    part_move: list[str]
    new_values: list[list[float]]

class Sequence(BaseModel):
    reasoning: str
    ordered_list: list[str]

class Approach(BaseModel):
    reasoning: str
    choice: int

class Translation3(BaseModel):
    list_translation: list[list[float]]

class Connection3(BaseModel):
    list_connection_type: list[str]

class Refinement(BaseModel):
    reasoning: str
    choice: list[int]
    type_choice: list[int]
    value_choice: list[float]

# Structure for GPT to predict the corresponding available objects, 6D transforms and color
class TransformArgs(BaseModel):
   part_names: list[str]
   available_part_correspondences: list[str]
   translation: list[list[float]]
   rotation: list[list[int]]
   color: list[list[int]]
   functional: list[bool]

# Predict to which part they are connected to (part pair) 
class PartPair(BaseModel):
   parts_pair: list[list[str]]
   connection_type: list[str]

# Predict the function calls and arguments - Maybe should be replaced by function calling
class FunctionPair(BaseModel):
   function_calls_name: list[str] #name of function to be called
   part_names: list[str] #argument

# Predict new parts and their corresponding object
class NewParts(BaseModel):
    list_part_class: list[str]
    list_avaiable_part_correspondence: list[str]
    number_instances: list[int]

class ReadjustRotation(BaseModel):
    part_names: list[str]
    rotation: list[list[int]]

class NewPartPair(BaseModel):
   parts_pair: list[list[str]]
   connection_type: list[str]
   boolean_operation: list[bool]
    
# Node class for connectivity tree
class Node:
    def __init__(self, name, parent=None, subnode=None):
        self.name = name
        self.parent=parent
        self.subnode_list = []
        self.bool_op = False
        
        if subnode:
            self.add_subnode(subnode)
        
        if parent:
            parent.add_subnode(self)
    
    def get_subnodes(self):
        return self.subnode_list
    
    def get_name(self):
        return self.name
    
    def add_subnode(self, subnode, bool_op):
        if not getNode(self, subnode.get_name()):
            self.subnode_list.append(subnode)
            subnode.parent = self
            if bool_op:
                self.bool_op = True
                subnode.bool_op = True
        else:
            print("subnode with this name exists elsewhere.")
    
    def subnode_exists(self):
        if self.subnode_list:
            return True
        else:
            return False

def getNode(parent_node:Node, name: str):
    if parent_node.get_name() == name:
        return parent_node

    if parent_node.subnode_exists():
        for subnode in parent_node.get_subnodes():
            exists = getNode(subnode, name)
            if exists:
                break
    else:
        return False
    return exists