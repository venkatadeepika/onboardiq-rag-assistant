import streamlit as st
import logging
from httpx import HTTPError
from ollama import ResponseError

class AssistantGUI:
    def __init__(self, assistant):
        self.assistant = assistant
        self.messages = assistant.messages
        self.employee_information = assistant.employee_information

    def get_response(self, user_input):
        return self.assistant.get_response(user_input)

    def render_messages(self):
        messages = self.messages

        for message in messages:
            if message["role"] == "user":
                st.chat_message("human").markdown(message["content"])
            if message["role"] == "ai":
                st.chat_message("ai").markdown(message["content"])

    def set_state(self, key, value):
        st.session_state[key] = value

    def render_employee_information(self):
        profile = self.employee_information
        st.subheader("Your onboarding profile")
        if not profile:
            st.info("No employee profile is available for this session.")
            return

        with st.container(border=True):
            st.caption("GETTING TO KNOW YOU")
            st.text(" ".join(
                profile[key] for key in ("name", "lastname") if profile.get(key)
            ))
            if profile.get("position"):
                st.text(profile["position"])

            for key, label in (
                ("department", "Your team"),
                ("location", "Your workplace"),
                ("supervisor", "Your go-to supervisor"),
                ("hire_date", "Your journey started"),
            ):
                if key in profile:
                    st.caption(label)
                    st.text(str(profile[key]))

        if profile.get("skills"):
            with st.expander("Your skills", expanded=True):
                for skill in profile["skills"]:
                    st.text(skill)

        with st.expander("How to reach you"):
            for key, label in (("email", "Email address"), ("phone_number", "Phone number")):
                if key in profile:
                    st.caption(label)
                    st.text(str(profile[key]))

        with st.expander("More profile details"):
            for key, label in (("employee_id", "Employee ID"), ("salary", "Salary")):
                if key in profile:
                    st.caption(label)
                    value = profile[key]
                    st.text(f"{value:,.2f}" if key == "salary" else str(value))

        st.caption("I'll use this profile to personalize your onboarding answers.")

    def render_user_input(self):
        user_input = st.chat_input("Ask me about your onboarding or company policies...", key="input")
        if user_input and user_input != "":
            st.chat_message("human").markdown(user_input)

            try:
                with st.chat_message("ai"):
                    response = st.write_stream(self.get_response(user_input))
            except (ConnectionError, HTTPError, ResponseError) as e:
                logging.error("Ollama response failed: %s", e)
                st.error(f"Ollama could not generate a response: {e}")
                return

            self.messages.append({"role": "user", "content": user_input})
            self.messages.append({"role": "ai", "content": response})

            self.set_state("messages", self.messages)

    def render(self):
        st.markdown(
            "<style>[data-testid=\"stToolbar\"] {display: none;}</style>",
            unsafe_allow_html=True,
        )
        with st.sidebar:
            st.logo(
                "https://upload.wikimedia.org/wikipedia/commons/0/0e/Umbrella_Corporation_logo.svg"
            )
            st.title("Umbrella Corporation Assistant")

            self.render_employee_information()

        self.render_messages()

        self.render_user_input()