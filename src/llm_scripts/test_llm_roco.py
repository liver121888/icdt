import openai

import sounddevice as sd
import numpy as np
from pynput import keyboard
import wave
import time
import subprocess

# Recording parameters
sample_rate = 16000  # Hz
channels = 2  # Stereo

recording = []
is_recording = False


openai.api_key = "sk-N3VbM3V2nZ4ZmOvWEBMByw"
openai.base_url = "https://cmu.litellm.ai"
client = openai.OpenAI(
  api_key="sk-N3VbM3V2nZ4ZmOvWEBMByw",
  base_url="https://cmu.litellm.ai",
)


def on_press(key):
    global is_recording
    if key == keyboard.Key.space and not is_recording:
        print("Recording started...")
        is_recording = True

def on_release(key):
    global is_recording, recording
    if key == keyboard.Key.space and is_recording:
        print("Recording stopped.")
        is_recording = False
        return False  # Stop listener

def record_audio():
    global recording, is_recording
    print("Press and hold the space bar to start recording.")
    print("Release the space bar to stop recording.")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while listener.running:
            if is_recording:
                audio_data = sd.rec(int(0.1 * sample_rate), samplerate=sample_rate, channels=channels)
                sd.wait()
                recording.append(audio_data)
        listener.join()

    # Concatenate all chunks and save to a .wav file
    if recording:
        recording = np.concatenate(recording, axis=0)
        file_name = f"TTR_prompt.wav"
        with wave.open(file_name, 'w') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())
    else:
        pass 



# dummy function to test proof of concept
def detect_objects():
    objects = {}
    objects['plate'] =  [372.0, 44.3, 585.52, 254.29]
    objects['can'] = [124.94, 114.74, 284.83, 219.25]
    objects['box'] = [676.8, 123.63, 853.57, 352.49]
    objects['ball'] = [277.68, 324.85, 357.17, 401.8]
    objects['block'] = [548.32, 369.17, 605.27, 429.57]
    return objects
def compute_centroid(x_min, y_min, x_max, y_max):
    return (x_min + x_max) / 2, (y_min + y_max) / 2
def move_gripper(x, y):
    print('moving gripper to location: ', x, y)


PATH_PLAN_INSTRUCTION="""
[Instruction]
You have access to the followings function: 
1) detect_objects() which returns a dictionary {object: (x_min, y_min, x_max, y_max)} where 'object' is a string
denoting the detected object, and (x_min, y_min, x_max, y_max) are the bounding box coordinates.
2) compute_centroid(x_min, y_min, x_max, y_max) which returns the centroid of the bounding box
2) move_gripper(x, y) which moves the gripper to the location (x,y,z)

Think step-by-step using comments, and make sure your response is executable as python code.
"""



if __name__=='__main__':
    formatted_string = "[Scene Description] You can observe the following objects and their corresponding bounding boxes:\n"
    objects = detect_objects()
    # Add entries from the dictionary
    for obj, bbox in objects.items():
        formatted_string += "{}: {}\n".format(obj, bbox)
    print(PATH_PLAN_INSTRUCTION)
    print(formatted_string)
    print("Describe the task for the robot to execute:")
    
    # Gather user input for the task description

    record_audio()

    result = subprocess.run(["./main", "-f", "TTR_prompt.wav" , "-np"], capture_output=True, text=True)
    task = result.stdout.split("]")[-1].strip()


    print('TASK', task)
    

    

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Or whichever model you're using
        messages=[
        {"role": "system", "content": formatted_string}, 
        {"role": "system", "content": PATH_PLAN_INSTRUCTION}, 
        {"role": "user", "content": task}],
        temperature=0, 
        max_tokens=500).to_dict()

    # Extract the code from the response
    generated_code = response['choices'][0]['message']['content']

    print("[Generated Code]\n", generated_code)
    exec(generated_code)


    print('[sanity check]')
    for object in objects:
        centroid = compute_centroid(*objects[object])
        print(object, centroid)