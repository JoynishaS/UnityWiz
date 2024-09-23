import streamlit as st
from llama_index.core import VectorStoreIndex,StorageContext,load_index_from_storage
from llama_index.vector_stores.milvus import MilusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

with st.sidebar:
    st.write("To chat with Unity Wiz ask them a question about Unity!")

st.header("UNITY WIZ!")
st.text_input("Ask me a question about Unity!")


#Configure settings for the application
Settings.text_splitter = SentenceSplitter(chunk_size=500)
Settings.embed_model = NVIDIAEmbedding(model = "NV-Embed-QA", truncate="END")
Settings.llm = NVIDIA(model = "meta/llama-3.1-70b-instruct")

index = None
query_engine = None