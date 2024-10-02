import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex,StorageContext, ServiceContext,load_index_from_storage,GPTVectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
import re

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
        st.session_state['query_engine'] = st.session_state['index'].as_query_engine(similarity_top_k=10,streaming=True,retriever_mode="embedding")

#Function to get chat responses
def stream_response(message):
    with st.chat_message("assistant",avatar="spade.jpeg"):
        st.session_state['results'] = st.empty()
        full_response = ""
        response_buffer = ""  # Buffer to accumulate text chunks

        #Send query with the last 3 chat interactions
        last_three = st.session_state['history'][-6:] if 'history' in st.session_state else []
        chat_history_revised = ""
        for item in last_three:
            if (item['role'] == 'assistant'):
                chat_history_revised +=(f"Original Answer:{item['content']}\n")

        # Combine the user's current message with the chat history for the model to understand the context
        query_input = chat_history_revised + f"\nNew Question:\nUser: {message}\nAssistant:"

        print(query_input)

        # Query the engine but only display the new response (not the query or history)
        st.session_state['response'] = st.session_state['query_engine'].query(query_input)

        # Stream the response and update the UI incrementally
        for text in st.session_state['response'].response_gen:
            response_buffer += text  # Collect the chunks in the buffer

            # Only process when we detect multiple 'Original Answer:'s
            if "Query:" in response_buffer or "Original Answer:" in response_buffer or '.' in response_buffer or "Rewrite:" in response_buffer :
                # Check if there are multiple 'Original Answer:' instances
                if response_buffer.count("Original Answer:") > 1:
                    # Handle the case of multiple original answers
                    response_buffer = remove_original_answers(response_buffer)

                # Filter out everything between Query: and Original Answer:
                response_buffer = filter_query(response_buffer)

                # Only append the assistant's filtered answer
                full_response += response_buffer
                response_buffer = ""  # Reset the buffer after processing

                # Update the UI with the latest filtered response
                st.session_state['results'].markdown(full_response)

        # Update session state with the new query and response pair
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append({'role': 'user', 'content': message})
        #Submit the fill history to the history variable
        st.session_state['history'].append({"role": "assistant", "content": full_response})


# Function to filter out everything between "Query:" and "Original Answer:"
def filter_query(response_text):
    # Use regex to remove all text between 'Query:' and 'Original Answer:' or 'Rewrite:'
    filtered_text = re.sub(r"Query:.*?(Original Answer:|Rewrite: |\?)", "", response_text, flags=re.DOTALL)

    # Remove the 'Original Answer:' or 'Rewrite:' tag itself if needed
    filtered_text = filtered_text.replace("Original Answer:", "").replace("Rewrite:", "").replace("?","")

    return filtered_text

# Function to handle multiple occurrences of 'Original Answer:'
def remove_original_answers(response_text):
    # This will handle the case where there are multiple "Original Answer:" occurrences
    # Here, you can decide how to deal with multiple original answers (keep only the last one, etc.)

    # Example: Keep only the last 'Original Answer:' and remove the others
    parts = response_text.split("Original Answer:")

    # Remove the query and previous original answers, only keep the last original answer
    if len(parts) > 1:
        response_text = "Original Answer:" + parts[-1]  # Keep only the last 'Original Answer:'


    # Clean up by removing the 'Original Answer:' text
    response_text = response_text.replace("Original Answer:", "")

    return response_text


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
            if message["role"] == "user":
                avatar ="smiley.png"
            else:
                avatar = "spade.jpeg"
            with st.chat_message(message["role"],avatar=avatar):
                st.markdown(message["content"])

    if user_input:
        with st.chat_message("user",avatar="smiley.png"):
            st.markdown(user_input)
        #st.session_state['history'].append({"role":"user","content":user_input})
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