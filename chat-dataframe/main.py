import streamlit as st
import pandas as pd
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import create_tool_calling_agent, AgentExecutor


# Streamlit page config
st.set_page_config(
    layout='wide',
    page_title="Chat DataFrame"
)
# Streamlit session state
if not 'messages' in st.session_state:
    st.session_state.messages = []

if not 'df' in st.session_state:
    st.session_state.df = None


#
#   LANGCHAIN
#
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
locals = {'df': st.session_state.df}
tools = [PythonAstREPLTool(locals=locals)]

llm_with_tools = llm.bind_tools(tools)

def get_response(query: str) -> str:
    global locals
    if st.session_state.df is None:
        return "Please upload a CSV file"
    
    # System prompt
    system = f"""Given a pandas dataframe `df` answer user's query.
    Here's the output of `df.head().to_markdown()` for your reference, you have access to full dataframe as `df`:
    ```
    {st.session_state.df.head().to_markdown()}
    ```
    Give final answer as soon as you have enough data, otherwise generate code using `df` and call required tool.
    If user asks you to make a graph, save it as `plot.png`, and output GRAPH:<graph title>.
    Example:
    ```
    plt.hist(df['Age'])
    plt.xlabel('Age')
    plt.ylabel('Count)
    plt.title('Age Histogram')
    plt.savefig('plot.png')
    ``` output: GRAPH:Age histogram
    Query:"""

    prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor.invoke({"input": query})['output']


# 
#   STREAMLIT APP
# 

st.title("Chat DataFrame")

# Sidebar for csv input
with st.sidebar:
    file = st.file_uploader("Upload CSV", type='csv')
    if file:
        st.session_state.df = pd.read_csv(file)

# Show messages above input box
for message in st.session_state.messages:
    st.chat_message(message[0]).write(message[1])

# Chat input
query = st.chat_input()
if query:

    # Append query message to messages
    st.session_state.messages.append(('user', query))
    with st.chat_message('user'):
        st.markdown(query)

    # Get ai response
    with st.chat_message('ai'):
        with st.spinner():
            res = get_response(query)

        if isinstance(res, pd.DataFrame):
            st.dataframe(res)
        elif res.find("GRAPH") != -1:
            # In case of graph, show it
            text = res[res.find("GRAPH")+6:]
            st.write(text)
            st.image('plot.png')
            st.session_state.messages.append(('ai', text))
        else:
            st.markdown(res)
            st.session_state.messages.append(('ai', res))