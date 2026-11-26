import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from ollama import ResponseError
from streamlit.testing.v1 import AppTest

from assistant import Assistant
from prompts import SYSTEM_PROMPT


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class TestVectorStore:
    def as_retriever(self):
        return RunnableLambda(lambda question: "Retrieved local policy")


class OllamaTests(unittest.TestCase):
    def test_assistant_streams_response(self):
        assistant = Assistant(
            system_prompt=SYSTEM_PROMPT,
            llm=FakeListChatModel(responses=["Local answer"]),
            vector_store=TestVectorStore(),
            employee_information={"name": "Test Employee"},
        )

        response = assistant.get_response("What is the leave policy?")

        self.assertNotIsInstance(response, str)
        self.assertEqual("".join(response), "Local answer")

    def test_unreachable_ollama_shows_setup_error(self):
        with patch.dict("sys.modules", {"langchain_chroma": Mock()}), patch(
            "ollama.Client", side_effect=ConnectionError("offline")
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertIn("Cannot connect to Ollama", app.error[0].value)
        self.assertIn("ollama serve", app.error[0].value)

    def test_missing_model_shows_pull_command(self):
        with patch.dict(os.environ, {"OLLAMA_CHAT_MODEL": "missing-chat"}), patch.dict(
            "sys.modules", {"langchain_chroma": Mock()}
        ):
            with patch("ollama.Client") as client:
                client.return_value.show.side_effect = ResponseError(
                    "model not found", status_code=404
                )
                app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertIn("ollama pull missing-chat", app.error[0].value)

    def test_vector_store_only_indexes_when_empty(self):
        for existing_ids in (["saved-policy"], []):
            with self.subTest(existing_ids=existing_ids):
                chroma_module = Mock()
                store = chroma_module.Chroma.return_value
                store.get.return_value = {"ids": existing_ids}
                with patch.dict("sys.modules", {"langchain_chroma": chroma_module}), patch(
                    "ollama.Client"
                ), patch(
                    "langchain_community.document_loaders.PyPDFLoader"
                ) as loader, patch("assistant.Assistant"), patch(
                    "gui.AssistantGUI"
                ), patch.dict(
                    os.environ,
                    {"OLLAMA_EMBEDDING_MODEL": f"test-embedding-{bool(existing_ids)}"},
                ):
                    loader.return_value.load.return_value = [
                        Document(page_content="Test policy")
                    ]
                    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                if existing_ids:
                    loader.assert_not_called()
                    store.add_documents.assert_not_called()
                else:
                    loader.assert_called_once()
                    store.add_documents.assert_called_once()
                    self.assertEqual(
                        store.add_documents.call_args.args[0][0].page_content,
                        "Test policy",
                    )

    def test_gui_displays_stream_once_and_saves_history(self):
        app = AppTest.from_string(
            """
import streamlit as st
from gui import AssistantGUI

class LocalAssistant:
    employee_information = {"name": "Test Employee"}

    def __init__(self):
        self.messages = st.session_state.get("messages", [])

    def get_response(self, question):
        return iter(["Local ", "answer"])

AssistantGUI(LocalAssistant()).render()
"""
        ).run()

        app.chat_input[0].set_value("What is the leave policy?").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            app.session_state["messages"],
            [
                {"role": "user", "content": "What is the leave policy?"},
                {"role": "ai", "content": "Local answer"},
            ],
        )
        self.assertEqual(
            [element.value for element in app.markdown].count("Local answer"), 1
        )

    def test_gui_renders_labeled_profile_without_json(self):
        app = AppTest.from_string(
            """
from gui import AssistantGUI

class LocalAssistant:
    messages = []
    employee_information = {
        "name": "Demo",
        "lastname": "Employee",
        "position": "Software Engineer",
        "department": "IT",
        "location": "Raccoon City HQ",
        "supervisor": "Demo Supervisor",
        "hire_date": "2026-01-01",
        "skills": ["Python", "Data Analysis"],
        "email": "demo@example.test",
        "phone_number": "555-0100",
        "employee_id": "demo-id",
        "salary": 65000.5,
    }

AssistantGUI(LocalAssistant()).render()
"""
        ).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.json), 0)
        self.assertIn("Your team", [element.value for element in app.caption])
        self.assertEqual(
            [element.value for element in app.text],
            [
                "Demo Employee", "Software Engineer", "IT", "Raccoon City HQ",
                "Demo Supervisor", "2026-01-01", "Python", "Data Analysis",
                "demo@example.test", "555-0100", "demo-id", "65,000.50",
            ],
        )
        self.assertEqual(
            [element.label for element in app.expander],
            ["Your skills", "How to reach you", "More profile details"],
        )

    def test_gui_missing_profile_shows_notice(self):
        app = AppTest.from_string(
            """
from gui import AssistantGUI

class LocalAssistant:
    messages = []
    employee_information = None

AssistantGUI(LocalAssistant()).render()
"""
        ).run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("No employee profile", app.info[0].value)


if __name__ == "__main__":
    unittest.main()
