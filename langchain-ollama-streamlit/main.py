from langchain_ollama.llms import OllamaLLM
import streamlit as st

#
#   LANGCHAIN OLLAMA WRAPPER
#

llm = OllamaLLM(model="llama3.2") # Using latest llama 3.2 3B

#
#   STREAMLIT APP
#

st.title("LLM App")

# User query input
query = st.text_input(label="Enter your query")

# Submit button
if st.button(label="Ask LLM", type="primary"):

    with st.container(border=True):
        with st.spinner(text="Generating response"):
            # Get response from llm
            response = llm.invoke(query)

        # Display it
        st.write(response)