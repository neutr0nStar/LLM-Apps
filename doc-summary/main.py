from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (Gemini API key)
load_dotenv()

#
#   Step 1: Load document content
#
def load_doc(file_bytes, file_name: str) -> str:
    
    # get file extension
    ext = file_name.split(".")[-1]
    save_file_name = "upload." + ext

    # save file locally
    with open(save_file_name, "wb") as f:
        f.write(file_bytes.getvalue())

    # Text files
    if ext == "txt":
        doc = TextLoader(save_file_name).load()
    elif ext == "pdf":
        doc = PyPDFLoader(save_file_name).load()

    doc = "\n\n".join([d.page_content for d in doc])
    return doc

#
#   Step 2: Summarise
#
def summarise(doc_content: str) -> str:

    # Load Gemini-1.5-flash model
    # API key is in the environment (.env file)
    model = GoogleGenerativeAI(model="gemini-1.5-flash")

    # prompt
    prompt = PromptTemplate.from_template("Provide a helpful summary of the contens provided by the user: {content}")

    # join prompt and model into a chain using LCEL
    summary_chain = prompt | model 

    # Any LLM has a fixed context window, or the number of tokens it can take in one run.
    # If our file is bigger than context window, we need to split it in chunks
    # and summarise chunks individually and then combine them for complete summary.
    
    # Here, we take it to be 4000 characters
    chunk_size = 4000 

    if len(doc_content) > chunk_size:
        
        # Split into chunks
        doc_chunks = []
        for i in range(len(doc_content) // chunk_size + 1):
            doc_chunks.append(
                doc_content[i : min(
                    i+chunk_size,
                    len(doc_content) # The last chunk may not necessarily contain 4000 characters
                    )]
                )
            
        # Summarise each chunk
        chunk_summary = []
        for chunk in doc_chunks:
            chunk_summary.append(summary_chain.invoke({"content": chunk}))

        # Now summarise all chunks combined
        summary = summary_chain.invoke({"content": "\n\n".join(chunk_summary)})
        return summary

    else:
        summary = summary_chain.invoke({"content": doc_content})
        return summary
    

# 
#   Streamlit server
# 

st.title("LLM Document Summariser")

file = st.file_uploader(label="Upload document", type=['pdf', 'txt'])
if file:
    with st.spinner("Summarising"):
        doc = load_doc(file, file_name=file.name)
        summary = summarise(doc)
    
    st.write("## Summary")
    st.write(summary)