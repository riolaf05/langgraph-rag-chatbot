from dotenv import load_dotenv
import streamlit as st
import platform
from my_graph import app
from pprint import pprint

# Variables ###############################################################
load_dotenv(override=True)
COLLECTION_NAME = "pdf"
FOLDER_PATH = r"C:\\Users\\ELAFACRB1\\Codice\GitHub\\langchain-document-chatbot\\documents" if platform.system()=='Windows' else '/documents'
MODEL_NAME="Llama3-8b-8192"
TEMPERATURE=0

st.title("Damian - Assistente turistico")
st.image('https://www.myrrha.it/wp-content/uploads/2018/06/via_marina_reggio.jpg', caption='Via Marina Reggio Calabria')
 
# Chatbot ##################################################################

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = MODEL_NAME

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("La tua domanda?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        inputs = {"question": prompt}
        try:
            for output in app.stream(inputs):
                for key, value in output.items():
                    pprint(f"Finished running: {key}:")
            pprint(value["generation"])
        except Exception as exc:
            print(exc)
            pprint("Non sono riuscito a comprendere la tua richiesta :(")                
        # #Create the stream 
        # stream = llm.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        # response = st.write_stream(stream)
        response = st.markdown(pprint(value["generation"]))
    
    st.session_state.messages.append({"role": "assistant", "content": value["generation"]})