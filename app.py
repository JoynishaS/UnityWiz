import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex,StorageContext,load_index_from_storage
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from time import sleep

#Configure settings for the application
Settings.text_splitter = SentenceSplitter(chunk_size=500)
Settings.embed_model = NVIDIAEmbedding(model = "NV-Embed-QA", truncate="END", api_key= st.secrets['NVIDIA_API_KEY'] )
Settings.llm = NVIDIA(model = "meta/llama-3.1-70b-instruct")

query_engine = None
progress = 0

def loadUnityDocumentation():
    st.session_state['initialized'] = True
    filePath = "lablab.pdf"
    index = None
    global progress
    global query_engine
    documents = []
    documents.extend(SimpleDirectoryReader(input_files=[filePath]).load_data())

    my_progress_bar = st.progress(0, "UnityWiz is loading up!")
    for i in range(100):
        sleep(0.1)  # Simulate a long-running task
         # Update the progress bar
        my_progress_bar.progress(i + 1, text= "Initializing UnityWiz!")

    if not documents:
        return f"There is no Unity Documentation PDF at this location"

    #Create a Milvus vector store and storage context
    vector_store = MilvusVectorStore(uri=st.secrets['ZILLZ_ENDPOINT_URI'],token=st.secrets['ZILLZ_API_KEY'],dim=1024,overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    #Create the index from the documents
    index = VectorStoreIndex.from_documents(documents,storage_context=storage_context)

    #Create the query engine
    query_engine = index.as_query_engine(similarity_top_k=20,streaming=True)
    st.success("Successfully loaded documents from files.")

#Function to handle chat interactions
def chat(message):
    global query_engine
    if query_engine is None:
        st.write( "UnityWiz is waiting for a question!")
    try:
        st.session_state['response'] = query_engine.query(message)
        st.write(message + st.session_state['response'])
    except Exception as e:
        st.error(f"Error proccessing query:{str(e)}")

#Function to stream responses
def stream_response(message):
    global query_engine
    if query_engine is None:
        st.write("UnityWiz is waiting for a question!")
    try:
        st.session_state['response'] = query_engine.query(message)
        st.session_state['partial_response'] = ""
        for text in st.session_state['response'].response_gen:
            st.session_state['partial_response'] += text
            st.write(message + st.session_state['partial_response'])
    except Exception as e:
        st.error(f"Error proccessing query:{str(e)}")

#Load Unity Documentation
if 'initialized' not in st.session_state:
    loadUnityDocumentation()

#interface
with st.sidebar:
    st.write("To chat with Unity Wiz ask them a question about Unity!")

st.header("UNITY WIZ!")
msg = st.text_input("Ask me a question about Unity!")
submit_btn = st.button("Submit")
clear_btn = st.button("Clear")
results = st.text_area("")

if submit_btn:
    results = stream_response(msg,chat(msg))

if clear_btn:
     results = ""