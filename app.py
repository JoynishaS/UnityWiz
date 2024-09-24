import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex,StorageContext,load_index_from_storage
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

#Configure settings for the application
Settings.text_splitter = SentenceSplitter(chunk_size=500)
Settings.embed_model = NVIDIAEmbedding(model = "NV-Embed-QA", truncate="END")
Settings.llm = NVIDIA(model = "meta/llama-3.1-70b-instruct")

query_engine = None

def loadUnityDocumentation():
    filePath = "UnityDocumentation.pdf"
    index = None
    global query_engine
    documents = []
    documents.extend(SimpleDirectoryReader(input_files=[filePath]).load_data())

    if not documents:
        return f"There is no Unity Documentation PDF at this location"

    #Create a Milvus vector store and storage context
    vector_store = MilvusVectorStore(uri="./milvus_unitywiz.db",dim=1024,overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    #Create the index from the documents
    index = VectorStoreIndex.from_documents(documents,storage_context=storage_context)

    #Create the query engine
    query_engine = index.as_query_engine(similarity_top_k=20,streaming=True)
    return f"Successfully loaded{len(documents)} documents from{len(filePath)} files."

#Function to handle chat interactions
def chat(message,history):
    global query_engine
    if query_engine is None:
        return history + [("UnityWiz is waiting for a question!",None)]
    try:
        response = query_engine.query(message)
        return history + [message,response]
    except Exception as e:
        return history + [(message, f"Error proccessing query:{str(e)}")]

#Function to stream responses
def stream_response(message, history):
    global query_engine
    if query_engine is None:
        yield history + [("UnityWiz is waiting for a question!",None)]
        return
    try:
        response = query_engine.query(message)
        partial_response = ""
        for text in response.response_gen:
            partial_response += text
            yield history +[(message,partial_response)]
    except Exception as e:
        yield history + [(message, f"Error proccessing query:{str(e)}")]

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

#Load Unity Documentation
loadUnityDocumentation()