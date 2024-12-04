import openai
import sounddevice as sd
import numpy as np
from pynput import keyboard
import wave
import time
import subprocess
import rclpy
from std_msgs.msg import String
from pynput.keyboard import Key, Listener

# Recording parameters
sample_rate = 16000  # Hz
channels = 2  # Stereo

devices = sd.query_devices()

# Print available devices to inspect the supported sample rates
for device in devices:
    print(device)

device = 11  # Use the index for sof-hda-dsp: - (hw:1,7)
recording = []
is_recording = False

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

    with Listener(on_press=on_press, on_release=on_release) as listener:
        while listener.running:
            if is_recording:
                audio_data = sd.rec(int(0.1 * sample_rate), samplerate=sample_rate, channels=channels, device=device)
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

def transcribe_audio():
    print('transcribing')
    result = subprocess.run(["/whisper.cpp/main", "-f", "TTR_prompt.wav" , "-np", "-m", "/home/kensuke/whisper.cpp/models/ggml-base.en.bin"], capture_output=True, text=True)
    task = result.stdout.split("]")[-1].strip()
    return task

def main():
    rclpy.init()  # Initialize the ROS client library

    # Create the ROS node
    node = rclpy.create_node('audio_transcription_node')

    # Create a publisher for the task string
    task_publisher = node.create_publisher(String, 'task_topic', 10)  # Publishing on 'task_topic'

    # Record audio
    print("Describe the task for the robot to execute:")
    record_audio()

    # Get the transcription result
    task = transcribe_audio()

    # Publish the transcription to the ROS topic
    task_msg = String()
    task_msg.data = task
    task_publisher.publish(task_msg)

    print(f"Published task: {task}")

    # Spin the ROS node to keep it running
    rclpy.spin(node)

    # Shutdown the ROS client library
    rclpy.shutdown()

if __name__ == '__main__':
    main()

