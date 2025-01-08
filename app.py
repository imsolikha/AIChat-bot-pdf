import streamlit as st
import base64
from vectors import Embeddings
from chatbot import ChatbotClass

def display(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f"<iframe src='data:application/pdf;base64,{base64_pdf}' width='100%' height='600' type='application/pdf'></iframe>"
    st.markdown(pdf_display, unsafe_allow_html=True)

# Initialize session state
if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.set_page_config(
    page_title="RAG Application",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar menu
with st.sidebar:
    st.markdown("Your personal PDF analyzer")
    menu = ['ChatBot RAG', 'MedicalBot']
    choice = st.selectbox("Models:", menu)

# Main application logic
if choice == "ChatBot RAG":
    st.header("Llama 3 RAG")
    col1, col2 = st.columns(2)

    with col1:
        st.header("🗃️ Upload your PDF")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            st.markdown(f"File name: {uploaded_file.name}")
            st.markdown(f"File size: {uploaded_file.size} bytes")

            temp_pdf_path = "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state['temp_pdf_path'] = temp_pdf_path

            # Display PDF preview
            st.markdown("Preview")
            display(uploaded_file)

            # Create embeddings
            create_embeddings = st.checkbox("Create Embeddings")
            if create_embeddings:
                try:
                    embeddings_manager = Embeddings(
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        qdrant_url="http://localhost:6333",
                        connection_name="chatbot_collection"
                    )
                    with st.spinner("Creating embeddings..."):
                        result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                    st.success(result)

                    if st.session_state['chatbot_manager'] is None:
                        st.session_state['chatbot_manager'] = ChatbotClass(
                            model_name="BAAI/bge-small-en",
                            device="cpu",
                            encode_kwargs={"normalize_embeddings": True},
                            llm_model="llama3",
                            temperature=0.7,
                            qdrant_url="http://localhost:6333",
                            collection_name="chatbot_collection"
                        )
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    with col2:
        st.header("Chat with PDF")

        if st.session_state['chatbot_manager'] is None:
            st.info("Please upload a PDF and create embeddings to start chatting.")
        else:
            for msg in st.session_state['messages']:
                st.chat_message(msg['role']).markdown(msg['content'])
            if user_input := st.chat_input("Type your message here..."):
                st.chat_message("user").markdown(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})

                with st.spinner("Answering..."):
                    try:
                        answer = st.session_state['chatbot_manager'].get_response(user_input)
                    except Exception as e:
                        answer = f"An error occurred while processing your request: {e}"

                st.chat_message("ChatBot").markdown(answer)
                st.session_state['messages'].append({"role": "ChatBot", "content": answer})
