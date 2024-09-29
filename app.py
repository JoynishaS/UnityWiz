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

def loadUnityDocumentation():
    st.set_page_config(layout="wide")
    st.session_state['initialized'] = True

    # Get Pre-existing Milvus vector store and storage context
    vector_store = MilvusVectorStore(uri=st.secrets['ZILLZ_ENDPOINT_URI'], token=st.secrets['ZILLZ_API_KEY'], dim=1024,
                                     collection_name="UnityDataCollection")

    # Create the indexed data from the vector_store
    if 'index' not in st.session_state:
        st.session_state['index'] = VectorStoreIndex.from_vector_store(vector_store)

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'query_engine' not in st.session_state:
        #Create the query engine
        st.session_state['query_engine'] = st.session_state['index'].as_query_engine(similarity_top_k=20,streaming=True)

#Function to handle chat interactions
def chat(message):
    try:
        st.session_state['response'] = st.session_state['query_engine'].query(message)
        stream_response(st.session_state['response'])
    except Exception as e:
        st.error(f"Error processing query:{str(e)}")

#Function to get chat responses
def stream_response(message):
    with st.chat_message("assistant"):
        st.session_state['results'] = st.empty()
        full_response = ""
        st.session_state['response'] = st.session_state['query_engine'].query(message)
        for text in st.session_state['response'].response_gen:
            full_response += text
            st.session_state['results'].markdown(full_response)
    st.session_state['history'].append({"role": "assistant", "content": full_response})


def main():
    #Load Unity Documentation
    if 'initialized' not in st.session_state:
        loadUnityDocumentation()

    #Side Bar Interface
    with st.sidebar:
        st.write("Chat with the Wiz to learn about Unity!")

    #Chat Bot Interface
    st.title("UNITY WIZ!")
    user_input = st.chat_input("Ask me a question about Unity!")

    #Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state['history']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state['history'].append({"role":"user","content":user_input})
        stream_response(user_input)

    #Move Clear Button to Bottom of Screen
    st.markdown(f"""
      <style>
      [class="st-emotion-cache-h23xut ef3psqc13"]{{
        display: flex;
        -webkit-box-align: center;
        align-items: center;
        -webkit-box-pack: center;
        justify-content: center;
        font-weight: 400;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        min-height: 2.5rem;
        margin-top: 5px;
        line-height: 1.6;
        color: inherit;
        width: auto;
        user-select: none;
        background-color: rgb(29, 27, 27);
        border: 1px solid rgba(241, 242, 245, 0.2);
        position: fixed;
        bottom:0;
        z-index:9999;
      }}
      </style>
    """, unsafe_allow_html=True)

    if st.button("Clear"):
        st.session_state['history'] = []
        st.rerun()


if __name__ == "__main__":
    main()