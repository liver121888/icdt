import streamlit as st
import ollama
import os, re

# Title of the Streamlit app
st.title("Ollama Chat Assistant")

# Initialize chat messages and selected model in session state if they don't exist
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role": "system", "content": "You are a helpful assistant."}]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3.2"

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant."

# Check if the selected model is available in Ollama by listing models using a shell command
def get_available_models():
    try:
        result = os.popen("ollama list").read()
        models = [line.strip() for line in result.splitlines()][2:]
        models = [model.split(" ")[0] for model in models]
        return models
    except Exception as e:
        st.error("Error retrieving model list")
        return []

# Dropdown for selecting a model
available_models = get_available_models()
model_name = st.selectbox("Choose a model:", available_models, index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0)

# Clear chat history if model is switched
if model_name != st.session_state.selected_model:
    st.session_state.selected_model = model_name
    st.session_state.chat_messages = [{"role": "system", "content": st.session_state.system_prompt}]

# Input for system prompt
system_prompt = st.text_input("System prompt:", value=st.session_state.system_prompt)

# Button to update the system prompt and clear history if desired
if st.button("Update System Prompt"):
    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
        st.session_state.chat_messages = [{"role": "system", "content": system_prompt}]

# Display the chat history
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to call Ollama's API and format assistant's response if it contains code
def get_response():
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.chat_messages]
    ollama_response = ollama.chat(model=st.session_state.selected_model, stream=True, messages=messages)

    assistant_message = ""
    for chunk in ollama_response:
        assistant_message += chunk["message"]["content"]
    
    # Check if response contains code using regex pattern for code blocks
    code_pattern = r"```(.*?)```"
    matches = re.findall(code_pattern, assistant_message, re.DOTALL)
    
    if matches:
        # Display the text before code, then code blocks in st.code
        parts = re.split(code_pattern, assistant_message)
        for i, part in enumerate(parts):
            if i % 2 == 0:
                st.markdown(part)  # Non-code text
            else:
                st.code(part, language="python")  # Code block
    else:
        st.markdown(assistant_message)
    
    return assistant_message

# Chat input at the bottom
if user_input := st.chat_input("Type your message here..."):
    # Append user's message to chat history
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            assistant_response = get_response()
                
    # Append assistant's message to chat history
    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})