import re, os, ast
import ollama
from openai import OpenAI

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


# SYSTEM_PROMPT = """
# You are a a robotic arm designed for spatial reasoning and positioning tasks. 
# You can observe scenes, analyze spatial relationships, and move to specified locations,
# Your end-effector doesn't have the capability to grasp objects but you can touch and push them.
# You have access to the following data:
#  - detections: a collection of detected objects in the scene with their centers/bboxes
# You have access to the following functions:
#  - robot_interface.move_robot (target_pose)
#  and standard Python libraries + functions.

#  Having selected a specific object, you can get its pose by calling robot_interface.get_pose(detected_object) method.

#  Provide the reasoning/your explantion followed by valid python snippet (without imports) after ```python.
#  Do not output anything after the code block.

# Here's a sample prompt and the expected response:
# [Prompt]
# Touch the box.

# [Output]
# Since I only need to move to the box I'm gonna set that to my target label and move to the detected object.

# ```python
# target_label = 'box'
# if target_label in detections:
#     detected_object = detections.find(target_label)
#     if detected_object:
#         print(f"{detected_object.label} found at {detected_object.center_3d}")
#         print(f"Moving to {detected_object.center_3d}")
#         robot_interface.move_robot(robot_interface.get_pose(detected_object))
# else:
#     print(f"{target_label} not found.")
# ```

# Here's the current scene description (For Reasoning if needed), followed by the new instruction:
# """

SYSTEM_PROMPT = """
You are a a robotic arm designed for spatial reasoning and positioning tasks. 
You can observe scenes, analyze spatial relationships, and move to specified locations,

You have access to the following data:
 - detections: a collection of detected objects in the scene with their centers/bboxes
You have access to the following functions:
 - robot_interface.move_robot (target_pose)
 - robot_interface.move_home()
 - robot_interface.get_pose (detected_object)
 - robot_interface.open_gripper()
 - robot_interface.close_gripper()
 and standard Python libraries + functions.

 Having selected a specific object, you can get its pose by calling robot_interface.get_pose(detected_object) method.
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


 

 0) import any python module if you need (but only standard libraries!). The robot_interface is already imported.
 1) Provide the reasoning/your explantion followed by valid python snippet (without imports) after ```python.
 2) Do not output anything after the code block.
 3) Always open the gripper before trying to close it for grasping.
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

Here's the current scene description (For Reasoning if needed), followed by the new instruction:
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
 - robot_interface.move_robot (target_pose)
 - robot_interface.move_home()
 - robot_interface.get_pose(detected_object)
 - robot_interface.sweep_left(obj_pose)
 - robot_interface.sweep_right(obj_pose)
 and standard Python libraries + functions.

 
 Having selected a specific object, you can get its pose by calling robot_interface.get_pose(detected_object) method.
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

 0) import any python module if you need (but only standard libraries!). The robot_interface is already imported.
 1) Provide the reasoning/your explantion followed by valid python snippet (without imports) after ```python.
 2) Do not output anything after the code block.
 3) DO NOT use `next` or any syntax like - detected_object = next((obj for obj in detections if obj.label == target_label), None)
 4) Instead, use the `find` method - detected_object = detections.find(target_label)
 5) There can only be one object with a given label.

Here's a sample prompt and the expected response:
[Prompt]
Sweep the can to the left.

[Output]
Since I need to sweep the can to the left, I'll call robot_interface.sweep_left(detected_object)

```python
target_label = 'can'
if target_label in detections:
    detected_object = detections.find(target_label)
    if detected_object:
        robot_interface.sweep_left(robot_interface.get_pose(detected_object))
else:
    print(f"{target_label} not found.")
```

Here's the current scene description (For Reasoning if needed), followed by the new instruction:
"""

if __name__ == "__main__":
    system_prompt = "You are Alice, a robotic arm designed for spatial reasoning and positioning tasks. You can observe scenes, analyze spatial relationships, and move to specified locations, but you cannot grasp or manipulate objects. Use built-in functions to support spatial analysis: compute_centroid(x_min: float, y_min: float, x_max: float, y_max: float) -> tuple: Calculates and returns the (x, y) centroid of a bounding box. move_gripper(x: float, y: float): Moves the gripper to specified (x, y) coordinates. Focus on analyzing the scene, selecting relevant objects, and moving to them within your capabilities. Provide step-by-step reasoning, and write executable Python code within a main function that accepts an objects list input."
    model_name = "llama3.1:latest"
    router = LLMRouter(model_name, system_prompt)
    user_prompt = """[Instruction]
    You have access to the followingsnary {object: (x_min, y_min, x_max, y_max)} where 'object' is a string
    denoting the detected object, and (x_min, y_min, x_max, y_max) are the bounding box coordinates.
    2) compute_centroid(x_min, y_min, x_max, y_max) which returns the centroid of the bounding box
    2) move_gripper(x, y) which moves the gripper to the location (x,y,z)

    Think step-by-step using comments, and make sure your response is executable as python code.

    [Scene Description] You can observe the following objects and their corresponding bounding boxes:
    plate: [372.0, 44.3, 585.52, 254.29]
    can: [124.94, 114.74, 284.83, 219.25]
    box: [676.8, 123.63, 853.57, 352.49]
    ball: [277.68, 324.85, 357.17, 401.8]
    block: [548.32, 369.17, 605.27, 429.57]

    Describe the task for the robot to execute:
    move to the any sports related object
    TASK move to the any sports related object
    """

    response = router(user_prompt)
