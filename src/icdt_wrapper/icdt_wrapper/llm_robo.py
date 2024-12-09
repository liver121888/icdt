import re, os, ast
import ollama
from openai import OpenAI
import re

OPENAI_KEY = "sk-eontrGWGppJx0343DEZl-A"


def old_parser(llm_output):
    # Split based on the first occurrence of '''
    parts = llm_output.split("```python", 1)
    # Everything before the ''' is considered reasoning
    reasoning = parts[0].strip()
    # Everything after the first ''' is considered code if it exists
    code = parts[1].strip() if len(parts) > 1 else None
    # Remove ``` from the beginning and the end of the code
    if code:
        code = code.strip("`").strip()
    print("here is the code:", code)
    # Check if code is valid
    valid_code = code is not None and len(code) > 0
    return reasoning, code if valid_code else None


def parser(llm_output):
    code_pattern = r"```([\s\S]*?)```"
    matches = re.findall(code_pattern, llm_output, re.DOTALL)

    if matches:
        # Display the text before code, then code blocks in st.code
        parts = re.split(code_pattern, llm_output)
        code, reasoning = [], []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                reasoning.append(part.strip())  # Non-code text
            else:
                if "python" in part[:10]:
                    part = part[6:]
                code.append(part)
        reasoning = "".join(reasoning)
        code = "".join(code)
        return reasoning, code
    else:
        return llm_output, None


def code_verify(code):
    # Parse the code into an Abstract Syntax Tree (AST)
    try:
        parsed_code = ast.parse(code)
        # Walk through the AST nodes and verify allowed nodes
        for node in ast.walk(parsed_code):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Code contains potentially unsafe import statements!")
            elif isinstance(node, ast.Call):
                # Check if the function name is 'eval' or 'exec'
                if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                    raise ValueError(
                        "Code contains potentially unsafe 'eval' or 'exec' calls!"
                    )

        # If no exception was raised, code is safe to execute
        exec(code)

    except ValueError as e:
        print(f"Code verification failed: {e}")


class OpenAILLM:
    def __init__(self, model_name, system_prompt):
        self.model_name = model_name
        self.system_prompt = system_prompt

    def __call__(self, prompt):
        response = openai_llm(self.model_name, self.system_prompt, prompt)
        return response.choices[0].message.content


def openai_llm(model_name, system_prompt, prompt):
    client = OpenAI(api_key=OPENAI_KEY, base_url="https://cmu.litellm.ai/")

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model=model_name,
    )
    return chat_completion


class OllamaLLM:
    def __init__(self, system_prompt, model_name):
        self.system_prompt = system_prompt
        self.model_name = model_name

    def __call__(self, prompt):
        response = ollama_llm(self.system_prompt, self.model_name, prompt)
        return response


def ollama_llm(system_prompt, model_name, prompt):
    def get_available_models():
        try:
            result = os.popen("ollama list").read()
            models = [line.strip() for line in result.splitlines()][2:]
            models = [model.split(" ")[0] for model in models]
            return models
        except Exception as e:
            print("Error retrieving model list")
            return []

    if model_name not in get_available_models():
        raise Exception("Model not available. Please choose another.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    ollama_response = ollama.chat(model=model_name, stream=True, messages=messages)
    assistant_message = ""
    for chunk in ollama_response:
        assistant_message += chunk["message"]["content"]
    return assistant_message


class LLMRouter:
    def __init__(self, model_name, system_prompt):
        if "gpt" in model_name:
            self.llm = OpenAILLM(model_name, system_prompt)
        else:
            self.llm = OllamaLLM(system_prompt, model_name)

    def __call__(self, user_prompt):
        response = self.llm(user_prompt)
        reasoning, code = parser(response)
        return reasoning, code


def parser_franka(prompt: str):
    # Match Franka's Prompt section regardless of order and handle variations in apostrophe/quote usage
    franka_section_pattern = r'<(?:Franka\'?s?|FRANKA\'?S?) ?Prompt>\s*(.*?)\s*(?:<(?:LBR\'?s?|LBR\'?S?) ?Prompt>|$)|<(?:LBR\'?s?|LBR\'?S?) ?Prompt>.*?<(?:Franka\'?s?|FRANKA\'?S?) ?Prompt>\s*(.*?)\s*$'
    
    franka_section = re.search(franka_section_pattern, prompt, re.DOTALL | re.IGNORECASE)
    
    if not franka_section:
        return None, None
        
    # Take the first non-None group (handles both ordering cases)
    franka_content = next((group for group in franka_section.groups() if group is not None), None)
    
    if not franka_content:
        return None, None
    
    # Extract Objective and Instructions from within this section
    objective_pattern = r'\*\*Objective\*\*:\s*(.+?)(?:\n|$)'
    instructions_pattern = r'\*\*Instructions for Franka\*\*:\s*((?:[-•]\s*.+\n?)+)'
    
    objective_match = re.search(objective_pattern, franka_content)
    instructions_match = re.search(instructions_pattern, franka_content)
    
    objective = objective_match.group(1).strip() if objective_match else None
    instructions = instructions_match.group(1).strip() if instructions_match else None
    
    return objective, instructions


def parser_lbr(prompt: str):
    # Match LBR's Prompt section regardless of order and handle variations in apostrophe/quote usage
    lbr_section_pattern = r'<(?:LBR\'?s?|LBR\'?S?) ?Prompt>\s*(.*?)\s*(?:<(?:Franka\'?s?|FRANKA\'?S?) ?Prompt>|$)|<(?:Franka\'?s?|FRANKA\'?S?) ?Prompt>.*?<(?:LBR\'?s?|LBR\'?S?) ?Prompt>\s*(.*?)\s*$'
    
    lbr_section = re.search(lbr_section_pattern, prompt, re.DOTALL | re.IGNORECASE)
    
    if not lbr_section:
        return None, None
        
    # Take the first non-None group (handles both ordering cases)
    lbr_content = next((group for group in lbr_section.groups() if group is not None), None)
    
    if not lbr_content:
        return None, None
    
    # Extract Objective and Instructions from within this section
    objective_pattern = r'\*\*Objective\*\*:\s*(.+?)(?:\n|$)'
    instructions_pattern = r'\*\*Instructions for LBR\*\*:\s*((?:[-•]\s*.+\n?)+)'
    
    objective_match = re.search(objective_pattern, lbr_content)
    instructions_match = re.search(instructions_pattern, lbr_content)
    
    objective = objective_match.group(1).strip() if objective_match else None
    instructions = instructions_match.group(1).strip() if instructions_match else None
    
    return objective, instructions



FRANKA_SYSTEM_PROMPT = """
You are a Franka robotic arm designed for spatial reasoning and positioning tasks. 
You can observe scenes, analyze spatial relationships, and move to specified locations,


You have access to the following data:
 - detections: a collection of detected objects in the scene with their centers/bboxes
You have access to the following functions:
 - franka_interface.move_robot (target_pose)
 - franka_interface.move_home()
 - franka_interface.get_pose (detected_object)
 - franka_interface.open_gripper()
 - franka_interface.close_gripper()
 and standard Python libraries + functions.

 Having selected a specific object, you can get its pose by calling franka_interface.get_pose(detected_object) method.
 Pose is a PoseStamped message with the following relevant fields:
 If we call it obj_pose, then:
 - obj_pose.pose.position.x
 - obj_pose.pose.position.y
 - obj_pose.pose.position.z
 - obj_pose.pose.orientation.x
 - obj_pose.pose.orientation.y
 - obj_pose.pose.orientation.z
 - obj_pose.pose.orientation.w

 So you can take a pose, add an offset to it, and move the robot to the new location.
 For this robot, the directions are as follows:
 - positive x | negative x: forward | backward
 - positive y | negative y: left | right
 - positive z | negative z: up | down
 So if you want to move 10cm forward, you would add 0.1 to position.x

 0) import any python module if you need (but only standard libraries!). The franka_interface is already imported.
 1) Provide the reasoning/your explantion followed by valid python snippet (without imports) after ```python.
 2) Do not output anything after the code block.
 3) Always open the gripper before trying to close it for grasping. Make sure the gripper is open even before you ever descend to the object.
 4) When you are told to "pick up an object", always move up a certain height above it. That is considered a proper "pick".
 5) Similarly, when you are told to stack/place/put something, always move up a certain height above it before releasing.
 6) When computing the pre-grasp and grasp poses, always apply the offsets to the original detected object pose (`obj_pose`), not to any modified poses like `pre_grasp_pose`. 

- **Pre-Grasp Pose**: To approach the object from above, add a positive offset to `obj_pose.pose.position.z`. For example, to move 10cm above the object, you would do:
  ```python
  pre_grasp_pose = copy.deepcopy(obj_pose)
  pre_grasp_pose.pose.position.z += 0.1
  ```
- **Grasp Pose**: To grasp the object properly, add a negative offset of 5cm along the z-axis to the original obj_pose. Do not adjust from pre_grasp_pose. For example:
  ```python
  grasp_pose = copy.deepcopy(obj_pose)
  grasp_pose.pose.position.z -= 0.05
  ```
- **Important**: Do not compute grasp_pose by modifying pre_grasp_pose. Always use a fresh copy of obj_pose when applying offsets for different poses.
- **Important**: These rules are only for grasping. For placing, you should always move up a certain height above the object before releasing.
- **Important**: ALWAYS USE FLOATS FOR ALL POSES.

+ **Important**: When searching for a detected object in `detections`, always use the `detections.find(label)` method.
+ Do not use loops or the `next` function.
+ Assume that `detections` has a `find(label)` method that returns the detected object with the given label.
+ There is only one object of each type, so you can assume `detections.find(label)` will return the correct object if it exists.
+ If there are multiple objects of the same type (let's say block in the scene description), ignore them. Those are false positives. 
Just use detections.find(label) to get the first one.

-------------------------------------------------------------
Here's a sample prompt and the expected response:
[Prompt]
Touch the box.

[Output]
Since I only need to move to the box I'm gonna set that to my target label and move to the detected object.

```python
target_label = 'box'
if target_label in detections:
    detected_object = detections.find(target_label)
    if detected_object:
        print(f"{detected_object.label} found at {detected_object.center_3d}")
        print(f"Moving to {detected_object.center_3d}")
        robot_interface.move_robot(robot_interface.get_pose(detected_object))
else:
    print(f"{target_label} not found.")
```

-------------------------------------------------------------
Here's the current scene description and the task to be performed:

"""

LBR_SYSTEM_PROMPT = """
You are a a robotic arm designed for spatial reasoning and positioning tasks. 
You can observe scenes, analyze spatial relationships, and move to specified locations.
You do not have a gripper, so you cannot grasp objects.
What you have is a flat surface to sweep/push objects.

You have access to the following data:
 - detections: a collection of detected objects in the scene with their centers/bboxes
 - given a detected object, you can get its width by calling detected_object.x_width / detected_object.y_width / detected_object.z_width
You have access to the following functions:
 - lbr_interface.move_robot (target_pose)
 - lbr_interface.move_home()
 - lbr_interface.get_pose(detected_object)
 - lbr_interface.sweep_left(obj_pose)
 - lbr_interface.sweep_right(obj_pose)
 and standard Python libraries + functions.

 
 Having selected a specific object, you can get its pose by calling lbr_interface.get_pose(detected_object) method.
 Pose is a PoseStamped message with the following relevant fields:
 If we call it obj_pose, then:
 - obj_pose.pose.position.x
 - obj_pose.pose.position.y
 - obj_pose.pose.position.z
 - obj_pose.pose.orientation.x
 - obj_pose.pose.orientation.y
 - obj_pose.pose.orientation.z
 - obj_pose.pose.orientation.w

 So you can take a pose, add an offset to it, and move the robot to the new location.

 For this robot, the directions are as follows:
 - positive x | negative x: backward | forward
 - positive y | negative y: right | left
 - positive z | negative z: up | down

 So if you want to move 10cm forward, you would subtract 0.1 from position.x

 0) import any python module if you need (but only standard libraries!). The lbr_interface is already imported.
 1) Provide the reasoning/your explantion followed by valid python snippet (without imports) after ```python.
 2) Do not output anything after the code block.
 5) There can only be one object with a given label.
 - **Important**: ALWAYS USE FLOATS FOR ALL POSES.
 - **Important**: DON'T MEMORIZE POSES. ALWAYS GET THE FRESHPOSE OF THE OBJECT FROM `detections.find(label)`


+ **Important**: When searching for a detected object in `detections`, always use the `detections.find(label)` method.
+ Do not use loops or the `next` function.
+ Assume that `detections` has a `find(label)` method that returns the detected object with the given label.
+ There is only one object of each type, so you can assume `detections.find(label)` will return the correct object if it exists.
+ If there are multiple objects of the same type (let's say block in the scene description), ignore them. Those are false positives. 
Just use detections.find(label) to get the first one.

Here's a sample prompt and the expected response:
[Prompt]
Sweep the can to the left.

[Output]
Since I need to sweep the can to the left, I'll call lbr_interface.sweep_left(detected_object)

```python
target_label = 'can'
if target_label in detections:
    detected_object = detections.find(target_label)
    if detected_object:
        lbr_interface.sweep_left(lbr_interface.get_pose(detected_object))
else:
    print(f"{target_label} not found.")
```

Here's the current scene description (For Reasoning if needed), followed by the new instruction:
"""



SYSTEM_PROMPT = f"""
You are a reasoning robot LLM responsible for high-level planning, reasoning, and delegating tasks to two worker LLMs: Franka and LBR. Each worker LLM represents a robot with distinct abilities:

1. **Franka (Gripper Robot)**:
   - Can move to specific positions in the workspace.
   - Can grasp objects using a gripper.
   - Can release objects.
   - CANNOT GO THE RECYCLE ZONE.
   - Left is +ve y and forward is +ve x.
   - This robot can only move within its workspace that is,
        - x is between -0.5 and -0.8
        - y is between 0.4 and -0.4
   - IT CANNOT MOVE THE END-EFFECTOR OUTSIDE THESE WORKSPACE LIMITS DUE TO SAFETY CONSTRAINTS. THAT IS YOU CANNOT GO TO A POSE LIKE `(-0.4, 0.0)`

2. **LBR (Sweeper Robot)**:
   - Has a sweeper as end-effector and cannot hold objects.
   - It can sweep objects in y direction.
   - Left is -ve y and forward is -ve x.
   - This robot can sweep an object if and only if the object is within its workspace that is,
        - x is between -0.3 and -0.5
        - y is between 0.4 and -0.4


Your task is to:
1. Analyze the high-level objective.
2. Reflect on your reasoning to ensure optimal task allocation.
3. Reason about the abilities of Franka and LBR to break down the task into exactly TWO subtasks to acheive the goal.
4. Provide prompt for each worker to give detailed explanations for each subtask.

**Important**: ALWAYS USE FLOATS FOR ALL POSES. So don't give a pose like `(-0.2,0)` for example. Give `(-0.2, 0.0)` instead.
**Important**: DON'T MEMORIZE POSES. ALWAYS GET THE FRESHPOSE OF THE OBJECT FROM `detections.find(label)`

Before outputting the anything output the sequence in the following format: 
Sequence = 1, 2 (1 for Franka and 2 for LBR). If only Franka does the task, then Sequence = 1. If only LBR does the task, then Sequence = 2.
If both robots are needed, then Sequence = 1, 2.
If none of the robots are needed, then Sequence = 0.
If both robots are needed but first LBR then Franka, then Sequence = 2, 1.

-------------------------------------------------------------
Example Task 1:

[Scene Description] You can observe the following objects and their corresponding center coordinates:
Can: [-0.6, 0.2, 0.07]
Cardboard_box: [-0.4, 0.2, 0.02]
Recycle Zone Bounding Box (X, Y values of top left and bottom right corners): [[-0.35, -0.2], [-0.5, -0.4]]
[Instruction] Move the can to the recycle zone.
[Output] 
Sequence = 1, 2

The objective is to move the can to the recycle zone. Given the capabilities of Franka and LBR, we can split the task into two subtasks:
1. **Franka's Subtask**: Grasp, lift, and place the can in succession to a location nearer to the recycle zone. The placement must be designed so that LBR can optimally perform its subsequent task.
2. **LBR's Subtask**: Sweep the can from its near position into the defined recycle zone fully.

<Franka's Prompt>
**Objective**: Grasp, move, and place both the can and the cardboard box to a designated drop-off area near the recycle zone.

**Instructions for Franka**:
 - Move to the current location of the can at coordinates `(-0.6, 0.2)`.
 - Grasp the can using your gripper.
 - Transport and place the can near the recycle zone, ensuring it is placed at `(-0.5, 0.0)`, a suitable position for LBR.


<LBR's Prompt>
**Objective**: Sweep the can into the recycle zone.

**Instructions for LBR**:
- Begin with the can:
  - Position yourself slightly right of where the can was placed by Franka at approximately.
  - You can get the updated can pose since detection is updated.
  - Sweep left to move the can completely into the recycle zone demarcated at `[-0.35, -0.2]`.

-------------------------------------------------------------
Example Task 2:

[Scene Description] You can observe the following objects and their corresponding center coordinates:
Can: [-0.3, 0.1]
Recycle Zone Bounding Box (X, Y corners): [[-0.35, -0.2], [-0.5, -0.4]]
[Instruction] Move the can to the recycle zone.
[Output] 
Sequence = 2

The objective is to move the can to the recycle zone. Given the capabilities of Franka and LBR, we can split the task into two subtasks:
1. **Franka's Subtask**: Franka doesn't need to do anything as LBR can directly sweep the can into the recycle zone.
2. **LBR's Subtask**: Sweep the can from its near position into the defined recycle zone fully.

<LBR's Prompt>
**Objective**: Sweep can into the recycle zone.

**Instructions for LBR**:
- Begin with the can:
  - Position yourself slightly right of where the can was placed by Franka at approximately.
  - You can get the updated can pose since detection is updated.
  - Sweep left to move the can completely into the recycle zone demarcated at `[-0.35, -0.2]`.

-------------------------------------------------------------
FOLLOW THE ABOVE TEMPLATE FOR ANSWERING.

Now solve this task:

"""

import re

def parse_task_output(task_output):
    # Extract the sequence
    sequence_match = re.search(r"Sequence = ([0-9, ]+)", task_output)
    sequence = list(map(int, sequence_match.group(1).split(','))) if sequence_match else []

    # Extract Franka's section
    franka_match = re.search(r"<Franka's Prompt>(.*?)(?:<LBR's Prompt>|$)", task_output, re.DOTALL)
    franka_section = franka_match.group(1).strip() if franka_match else ""

    # Extract LBR's section
    lbr_match = re.search(r"<LBR's Prompt>(.*)", task_output, re.DOTALL)
    lbr_section = lbr_match.group(1).strip() if lbr_match else ""

    # Construct the result dictionary
    result = {
        "sequence": sequence,
        "instructions": {
            "franka": franka_section,
            "lbr": lbr_section
        }
    }

    return result
    

def run_llm():
    model_name = "gpt-4o"
    reasoning_router = LLMRouter(model_name, SYSTEM_PROMPT)
    user_prompt = """
[Scene Description] You can observe the following objects and their corresponding bounding boxes:
Can: [-0.5, 0.4, 0.07]
Recycle zone bounding box: [[-0.2, -0.4, 0], [-0.4, -0.4, 0], [-0.4, -0.2, 0], [-0.2, -0.2, 0]]
    """
    response = reasoning_router(user_prompt + '\n[Instruction] Move the recyclable objects to the recycle zone.')
    print(response[0])
    print("*"*50)

    r1, r2 = parser_franka(response[0])
    tmp = f"""
[Prompt] {r1}
**Instructions for Franka**:
{r2}

[Output]
"""
    franka_router = LLMRouter(model_name, FRANKA_SYSTEM_PROMPT)
    franka_response = franka_router(user_prompt+tmp)
    print('\n'.join(list(franka_response)))    
    return franka_response