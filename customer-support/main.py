import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass
from typing import Literal, List
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool


@dataclass
class Order:
    id: int
    name: str
    status: Literal["placed", "delivered", "cancelled"]


class OrdersTool:

    def __init__(self):

        # Dummy data, use database in prod
        self.orders: List[Order] = [
            Order(id=1, name="Washine Machine", status="placed"),
            Order(id=2, name="LCD TV", status="delivered"),
            Order(id=3, name="Fridge", status="cancelled"),
        ]

    def get_all(self) -> List[Order]:
        """Get all orders"""
        return [vars(o) for o in self.orders]

    def get_latest(self) -> Order:
        """Get last order from the list"""
        return vars(self.orders[-1])


class RetrievalTool:
    """Retrieve and load data in vectorstore"""

    def __init__(self, kb_file_path: str):
        self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma(
            collection_name="knowledge_base",
            embedding_function=self.embedding_function,
        )
        self.kb_file_path = kb_file_path

    def init(self):
        """Load knolwedge from md file"""
        # Loader
        loader = TextLoader(file_path=self.kb_file_path)
        docs = loader.load()

        # Splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Load to db
        self.vectorstore.add_documents(chunks)

    def retrieve(self, query: str):
        """Retrieve relevant chunks from vectorstore"""
        return "\n\n".join(
            doc.page_content for doc in self.vectorstore.similarity_search(query, k=3)
        )


class TicketTool:
    """A simple tool to raise a new ticket"""

    def raise_new(self, contents: str) -> str:
        """Raise a new ticket"""
        # In prod, this will be connected to some ticket raising service
        print("Ticket raised")
        return "Ticket successfully registered"


class CustomerSupportAgent:

    def __init__(self, tools: List):
        load_dotenv()
        llm = ChatOpenAI(
            model="moonshotai/kimi-k2:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        self.agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt="You are a helpful customer support agent. Answer the user queries with the help of given tools",
        )

    def run(self, messages: List) -> str:
        """Invoke the agent with given messages"""
        return self.agent.invoke({"messages": messages})["messages"][-1].content


class App:
    """Streamlit App"""

    def __init__(self, customer_support_agent: CustomerSupportAgent):
        self.customer_support_agent = customer_support_agent

    def run(self):
        # Session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Title
        st.title("Customer support app")

        # Chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input():

            # Echo user input
            with st.chat_message("user"):
                st.markdown(prompt)

            # Append to messages
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get ai response
            with st.spinner():
                ai_response = self.customer_support_agent.run(st.session_state.messages)

            # Show output
            with st.chat_message("ai"):
                st.markdown(ai_response)

            # Append to messages
            st.session_state.messages.append({"role": "ai", "content": ai_response})


if __name__ == "__main__":
    # Tools
    orders_tool = OrdersTool()
    retrieval_tool = RetrievalTool("./kb.md")
    retrieval_tool.init()
    ticket_tool = TicketTool()

    @tool
    def get_all_orders():
        """Get all orders"""
        return orders_tool.get_all()

    @tool
    def get_latest_order(inp: str):
        """Get latest order"""
        return orders_tool.get_latest()

    @tool
    def retrieve_from_kb(query: str):
        """Retrieve relevant data from knowledge base"""
        return retrieval_tool.retrieve(query)

    @tool
    def raise_ticket(contents: str):
        """Raise a new complain ticket"""
        return ticket_tool.raise_new(contents)

    csa = CustomerSupportAgent(
        tools=[get_all_orders, get_latest_order, retrieve_from_kb, raise_ticket]
    )
    app = App(customer_support_agent=csa)
    app.run()
