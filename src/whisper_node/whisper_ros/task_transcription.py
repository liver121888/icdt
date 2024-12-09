import openai
import sounddevice as sd
import numpy as np
from pynput import keyboard
import wave
import time
import os
import subprocess
import rclpy
from std_msgs.msg import String
from pynput.keyboard import Key, Listener
from scipy.io import wavfile
from scipy.signal import resample

# Recording parameters

devices = sd.query_devices()

# Print available devices to inspect the supported sample rates
# Assuming 'devices' is a list of dictionaries containing device info

target_device = {}
for device in devices:
    print(device)

    # Check if the target_device exists
    if 'CMTECK' in device['name']:
        target_device = device

if target_device['index'] != -1:
    print(f"Device index for target_device: {target_device['index']}")
else:
    print("target_device not found.")
    exit()

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

    # Audio parameters
    input_channels = 1

    def audio_callback(indata, frames, time, status):
        if is_recording:
            recording.append(indata.copy())  # Append each audio chunk
    with sd.InputStream(samplerate=target_device['default_samplerate'], channels=input_channels, device=target_device['index'], callback=audio_callback):
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    if recording:
        # Concatenate all audio chunks and save as a .wav file
        audio_data = np.concatenate(recording, axis=0)
        print(f"Audio data range: {audio_data.min()} to {audio_data.max()}")

        # Convert float32 to int16 for WAV file
        audio_data = (audio_data * 32767).astype(np.int16)
        file_name = "TTR_prompt.wav"

        # Save audio to a WAV file
        with wave.open(file_name, 'w') as wf:
            wf.setnchannels(input_channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(target_device['default_samplerate'])
            wf.writeframes(audio_data.tobytes())
        
        print(f"Audio recorded and saved as {file_name}")
    else:
        print("No audio recorded.")

def transcribe_audio():
    # Read the input .wav file
    rate, data = wavfile.read("TTR_prompt.wav")

    # Define the new sample rate
    new_rate = 16000

    # Calculate the number of samples for the new rate
    new_num_samples = int(len(data) * new_rate / rate)

    # Resample the data
    resampled_data = resample(data, new_num_samples)

    # Save the resampled audio
    wavfile.write("output.wav", new_rate, resampled_data.astype(data.dtype))

    print('resampled')

    result = subprocess.run(["/whispercpp/main", "-f", "output.wav" , "-np", "-m", "/whispercpp/models/ggml-base.en.bin"], capture_output=True, text=True)
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
