
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from httpx import HTTPError
from ollama import Client, ResponseError
import os
from assistant import Assistant
from gui import AssistantGUI
from prompts import SYSTEM_PROMPT, WELCOME_MESSAGE

from data.employees import generate_employee_data

if __name__ == "__main__":

    load_dotenv()

    st.set_page_config(
        page_title="Umbrella Onboarding",
        page_icon=":umbrella:",
        layout="wide"
    )

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    chat_model = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:1.5b").strip()
    embedding_model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text").strip()
    if not all((base_url, chat_model, embedding_model)):
        st.error("OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL and OLLAMA_EMBEDDING_MODEL cannot be blank.")
        st.stop()

    try:
        client = Client(host=base_url, timeout=10)
        for model in (chat_model, embedding_model):
            client.show(model)
    except ResponseError as e:
        if e.status_code == 404:
            st.error(f"Ollama model '{model}' is not installed. Run: ollama pull {model}")
        else:
            st.error(f"Ollama model check failed: {e}")
        st.stop()
    except (ConnectionError, HTTPError) as e:
        st.error(f"Cannot connect to Ollama at {base_url}. Start the Ollama app or run: ollama serve")
        st.stop()

    @st.cache_data(ttl=3600, show_spinner="Generating employee data...")
    def generate_data():
        return generate_employee_data(1)[0]

    @st.cache_resource(ttl=3600, show_spinner="Initializing vector store...")
    def init_vector_store(pdf_path, base_url, embedding_model):

        try:
            embedding_function = OllamaEmbeddings(
                model=embedding_model,
                base_url=base_url,
            )
            persistent_path = r"data\vectorstore_ollama"

            vector_store = Chroma(
                embedding_function=embedding_function,
                persist_directory=persistent_path,
            )
            if not vector_store.get(limit=1)["ids"]:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(documents)
                vector_store.add_documents(splits)
            return vector_store
        except (ConnectionError, HTTPError, ResponseError) as e:
            st.error(
                "Ollama could not generate embeddings. Quit and reopen the Ollama app, "
                "then rerun Streamlit. Do not change the temporary /tokenize port "
                "or delete the saved vector store. "
                f"Details: {e}"
            )
            return None
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            return None

    customer_data = generate_data()
    vector_store = init_vector_store(
        r"data\umbrella_corp_policies.pdf", base_url, embedding_model
    )
    if vector_store is None:
        st.stop()

    if "customer" not in st.session_state:
        st.session_state.customer = customer_data
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": WELCOME_MESSAGE}]

    llm = ChatOllama(
        model=chat_model,
        base_url=base_url,
        temperature=0,
    )

    assistant = Assistant(
        system_prompt=SYSTEM_PROMPT,
        llm=llm,
        message_history=st.session_state.messages,
        vector_store=vector_store,
        employee_information=st.session_state.customer
    )

    gui = AssistantGUI(assistant)
    gui.render()