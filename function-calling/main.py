from langchain_ollama.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

# 
#   LANGCHAIN FUNCTION CALLING
# 

@tool
def get_weather(city: str):
    """Fetches weather for given city"""
    weather_data = { # Example data, use API in real-world
        "New York": "Sunny, 25°C", 
        "London": "Cloudy, 18°C", 
        "Tokyo": "Rainy, 22°C"
        }
    return weather_data.get(city, "Weather data not available.")


def get_response(prompt: str, use_function_calling: bool):
    messages = [
        SystemMessage("You are a helpful AI assistant. Answer concisely."),
        HumanMessage(prompt)
    ]
    if use_function_calling:
        model = ChatOllama(model="llama3.2").bind_tools([get_weather])
        res = model.invoke(messages)
        messages.append(res)

        for tool_call in res.tool_calls:
            selected_tool = {"get_weather": get_weather}[tool_call['name'].lower()]
            tool_message = selected_tool.invoke(tool_call)
            messages.append(tool_message)

        res = model.invoke(messages)
        messages.append(res)

        return res.content, messages

    else:
        model = ChatOllama(model="llama3.2")
        res = model.invoke(messages)
        messages.append(res)
        return res.content, messages

    

#
#   STREAMLIT
#
import streamlit as st

st.title("Function Calling in LLMs")

# Toggle to turn on function calling
use_function_calling = st.toggle("Turn on function calling for weather")

# Prompt input
prompt = st.text_input("Prompt")

# Submit button
if st.button("Submit", type="primary", use_container_width=True):
    with st.spinner():
        res, messages = get_response(prompt, use_function_calling)

        st.write(res)

        with st.expander("Chat messages"):
            st.write(messages)