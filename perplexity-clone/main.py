import os
import requests
from typing import Iterator, List

from langchain_openai.chat_models import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-oss-20b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

search = GoogleSerperAPIWrapper()

prompt = ChatPromptTemplate.from_template(
    "Answer based on the content of web pages provided to you.\n"
    "Question: {question}\n"
    "Page contents: {page_contents}"
)


def get_web_results(query: str) -> List[str]:
    try:
        results = search.results(query)
    except Exception:
        return []

    organic = results.get("organic", []) if isinstance(results, dict) else []
    return [item.get("link", "") for item in organic[:3] if item.get("link")]


def fetch_page_content(link: str) -> str:
    try:
        res = requests.get(f"https://r.jina.ai/{link}", timeout=15)
        res.raise_for_status()
        return res.content.decode(errors="ignore")[:1000]
    except Exception:
        return ""


def stream_answer(query: str, page_contents: List[str]) -> Iterator[str]:
    messages = prompt.format_messages(
        question=query, page_contents="\n\n".join(page_contents)
    )
    for chunk in llm.stream(messages):
        text = chunk.content or ""
        if not text and isinstance(chunk.content, list):
            text = "".join(
                part.get("text", "")
                for part in chunk.content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if text:
            yield text


def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Perplexity Clone", page_icon="🔎")
    st.title("Perplexity Clone")
    st.caption("Ask a question and watch the research unfold step by step.")

    with st.form("query_form"):
        query = st.text_input("Your question")
        submitted = st.form_submit_button("Research")

    if not submitted or not query:
        return

    status_area = st.empty()
    sources_placeholder = st.empty()
    answer_placeholder = st.empty()

    status_area.info("Searching the web...")
    links = get_web_results(query)

    if links:
        with sources_placeholder.container():
            st.subheader("Sources")
            for link in links:
                st.markdown(f"- [{link}]({link})")
        status_area.success(f"Found {len(links)} source(s).")
    else:
        status_area.warning("No links found for this query. Please try a different question.")
        return

    status_area.info("Fetching page contents...")
    progress = st.progress(0.0, text="Fetching page contents...")
    page_contents: List[str] = []
    for idx, link in enumerate(links, start=1):
        content = fetch_page_content(link)
        if content:
            page_contents.append(content)
        status_area.info(f"Fetching page contents... ({idx}/{len(links)})")
        progress.progress(idx / len(links), text=f"Fetched {idx} of {len(links)} pages...")
    progress.empty()

    if not page_contents:
        status_area.error("Unable to retrieve page contents. Please try again later.")
        return

    status_area.info("Generating answer...")

    accumulated = ""
    for chunk in stream_answer(query, page_contents):
        accumulated += chunk
        answer_placeholder.markdown(accumulated)

    status_area.success("Answer ready.")


if __name__ == "__main__":
    run_streamlit_app()
