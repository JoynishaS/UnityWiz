import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex,StorageContext, ServiceContext,load_index_from_storage,GPTVectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

#Configure settings for the application
Settings.text_splitter = SentenceSplitter(chunk_size=500,chunk_overlap=20)
Settings.embed_model = NVIDIAEmbedding(model = "NV-Embed-QA", truncate="END", api_key= st.secrets['NVIDIA_API_KEY'] )
Settings.llm = NVIDIA(model = "meta/llama-3.2-3b-instruct")

index = None
progress = 0

def loadUnityDocumentation():
    st.session_state['initialized'] = True

    # Get Pre-existing Milvus vector store and storage context
    vector_store = MilvusVectorStore(uri=st.secrets['ZILLZ_ENDPOINT_URI'], token=st.secrets['ZILLZ_API_KEY'], dim=1024,
                                     collection_name="UnityDataCollection")

    # Create the indexed data from the vector_store
    index = VectorStoreIndex.from_vector_store(vector_store)

    if 'query_engine' not in st.session_state:
        #Create the query engine
        st.session_state['query_engine'] = index.as_query_engine(similarity_top_k=20,streaming=True)
        st.success("Successfully loaded documents from files.")

#Function to handle chat interactions
def chat(message):
    if st.session_state['query_engine'] is None:
        st.write( "UnityWiz is waiting for a question!")
    try:
        st.session_state['response'] = st.session_state['query_engine'].query(message)
        stream_response(st.session_state['response'])
    except Exception as e:
        st.error(f"Error proccessing query:{str(e)}")

#Function to stream responses
def stream_response(message):
    if st.session_state['query_engine'] is None:
        st.write("UnityWiz is waiting for a question!")
    try:
        st.session_state['response'] = st.session_state['query_engine'].query(message)
        full_response = ""
        for text in st.session_state['response'].response_gen:
            full_response += text
            st.session_state['full_response'] = full_response
    except Exception as e:
        st.error(f"Error proccessing query:{str(e)}")

def updateTextArea():
    response = st.session_state['full_response']

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


if submit_btn:
    stream_response(msg)
    results = st.text_area(st.session_state['full_response'])

if clear_btn:
     results = ""