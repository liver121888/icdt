# icdt



# For whisper
cd whisper.cpp
sh ./models/download-ggml-model.sh tiny.en
make -j

# in test_llm_roco.py
You will asked to hold the spacebar down to record.
This saves a file 'TTR_prompt.wav' file
We then call subprocess.run to allow whisper to transcribe the .wav file and return the command as a string
LLM inference as normal