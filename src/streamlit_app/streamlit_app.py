import os

import requests
import streamlit as st
from streamlit_chat import message


class MovieStreamlitApp:
    URL = "http://localhost:5600/generate"
    def __init__(self):
        self.build_app()
        self.initialize_session()
        self.clear_button = st.sidebar.button("Clear Conversation", key="clear")

    @staticmethod
    def initialize_session():
        """"""
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        if 'past' not in st.session_state:
            st.session_state['past'] = []
        if 'messages' not in st.session_state:
            st.session_state['messages'] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
        if 'model_name' not in st.session_state:
            st.session_state['model_name'] = []
        if 'cost' not in st.session_state:
            st.session_state['cost'] = []
        if 'total_tokens' not in st.session_state:
            st.session_state['total_tokens'] = []
        if 'total_cost' not in st.session_state:
            st.session_state['total_cost'] = 0.0

    @staticmethod
    def build_app():
        st.set_page_config(page_title="BOB", page_icon=":robot_face:")
        st.markdown("<h1 style='text-align: center;'>Book Writing Bot</h1>", unsafe_allow_html=True)

    @staticmethod
    def generate_response(prompt):
        st.session_state['messages'].append({"role": "user", "content": prompt})

        json_obj = {"prompt": prompt}
        response = requests.post(MovieStreamlitApp.URL, json=json_obj)
        try:
            generated_text = response.json().get("response")
        except:
            return "Looks like the response was empty. Try again."

        st.session_state['messages'].append({"role": "assistant", "content": generated_text})
        return generated_text


if __name__ == '__main__':

    ms = MovieStreamlitApp()

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()
    if ms.clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = MovieStreamlitApp.generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
