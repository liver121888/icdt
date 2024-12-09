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
        code = code.strip('`').strip()
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
                if 'python' in part[:10]:
                    part = part[6:]
                code.append(part)
        reasoning = ''.join(reasoning)
        code = ''.join(code)
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
                    raise ValueError("Code contains potentially unsafe 'eval' or 'exec' calls!")

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
    
    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    ollama_response = ollama.chat(model=model_name, stream=True, messages=messages)
    assistant_message = ""
    for chunk in ollama_response:
        assistant_message += chunk["message"]["content"]
    return assistant_message


class LLMRouter:
    def __init__(self, model_name, system_prompt):
        if 'gpt' in model_name:
            self.llm = OpenAILLM(model_name, system_prompt)
        else:
            self.llm = OllamaLLM(system_prompt, model_name)

    def __call__(self, user_prompt):
        response = self.llm(user_prompt)
        print(response)
        print('$'*50)
        reasoning, code = parser(response)
        print(reasoning)
        print('$'*50)
        print(code)
        # print(reasoning)
        # code_verify(code)
        #exec(code)
        #return reasoning, code


if __name__ == '__main__':
    system_prompt = "You are a helpful assistant."  
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