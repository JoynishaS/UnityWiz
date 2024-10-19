import streamlit as st
from llama_index.core import  VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
import re
import requests


#Configure settings for the application
Settings.text_splitter = SentenceSplitter(chunk_size=1000,chunk_overlap=20)
Settings.embed_model = NVIDIAEmbedding(model = "NV-Embed-QA", truncate="END", api_key= st.secrets['NVIDIA_API_KEY'] )


def loadUnityDocumentation():
    st.set_page_config(layout="wide")
    st.session_state['initialized'] = True

    # Get pre-existing Milvus vector store and set up storage context
    vector_store = MilvusVectorStore(
        uri=st.secrets['ZILLZ_ENDPOINT_URI'],
        token=st.secrets['ZILLZ_API_KEY'],
        dim=1024,
        collection_name="UnityDataCleanCollection"
    )

    # Create the indexed data from the vector_store if not already present
    if 'index' not in st.session_state:
        st.session_state['index'] = VectorStoreIndex.from_vector_store(vector_store)

    # Initialize the chat history and query engine
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'query_engine' not in st.session_state:
        # Create the query engine
        st.session_state['query_engine'] = st.session_state['index'].as_query_engine(
            similarity_top_k=10, streaming=True, retriever_mode="embedding"
        )

def check_for_token_id(generated_text, token_id=128009):
    url = "http://34.95.153.148:8000/check_token_id"  # Flask server endpoint
    payload = {
        "generated_text": generated_text,
        "token_id": token_id
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        tokens = result.get("tokens", [])
        print(f"Token IDs for the generated text: {tokens}")
        print(result)
        if result.get('token_present'):
            print(f"Token ID {token_id} is present in the generated text.")
            return True
        else:
            print(f"Token ID {token_id} is not present in the generated text.")
    else:
        print(f"Error in checking token ID: {response.status_code}")

# Function to generate a response via Flask API on Google Cloud GPU
def generate_response(question,continuation_text="Continue:", max_rounds=5, context_window=300):
    full_generated_text = ""
    global new_context
    new_context = ""
    current_context = question  # Start with the question only
    rounds = 0

    while rounds < max_rounds:
        # Send request to your Flask API running on the GPU instance
        url = "http://34.95.153.148:8000/generate"   # Replace with actual server IP
        payload = {"prompt": current_context}
        response = requests.post(url, json=payload)

        # Handle the API response
        if response.status_code == 200:
            generated_text = response.json().get("generated_text", "")
            full_generated_text += generated_text

            # Check if end of the generation is reached
            if check_for_token_id(full_generated_text, 128009):  # Assuming 128009 is the EOD token
                break

            # Avoid repetition by trimming the previously generated text, keeping only the last part for context
            new_context = (full_generated_text)[-context_window:]  # Keep only the last part of the answer
            current_context = "finish answering the question from this context" + new_context  # Set continuation hint

            rounds += 1
        else:
            return "Error in generation."

    return full_generated_text


def stream_response(message):
    with st.chat_message("assistant", avatar="spade.jpeg"):
        st.session_state['results'] = st.empty()
        full_response = ""
        response_buffer = ""  # Buffer to accumulate text chunks

        # Prepare the last 3 interactions for context
        last_three = st.session_state['history'][-6:] if 'history' in st.session_state else []
        chat_history_revised = ""
        for item in last_three:
            if item['role'] == 'assistant':
                chat_history_revised += f"Original Answer: {item['content']}\n"

        # Combine current message with chat history
        query_input = chat_history_revised + f"\nNew Question:\nUser: {message}\nAssistant:"

        # Perform a query with the query engine and handle streaming output
        st.session_state['response'] = st.session_state['query_engine'].query(query_input)

        # Collect all chunks in a list and join them with spaces
        chunks = [chunk.strip() for chunk in st.session_state['response'].response_gen]
        search_output = " ".join(chunks) + " "

        # Remove any multiple spaces that may have accumulated
        search_output = re.sub(r'\s+', ' ', search_output).strip()

        print(search_output)

        prompt = f"""
        You are an expert Unity assistant who provides technical explanations with relevant Unity code snippets where applicable.
        Provide a detailed technical answer to the question based on the following context:
        Question: {message}
        Context: {search_output}
        Include relevant Unity code snippets if applicable.

        Answer:
        """
        generated_text = generate_response(prompt)

        # Find the start of the assistant's answer
        answer_start = generated_text.find("Answer:")
        if answer_start != -1:
            # Remove everything up to the assistant's response
            final_answer = generated_text[answer_start + len("Answer:"):].strip()
        else:
            # If the "assistant:" label isn't found, just show the entire response
            final_answer = generated_text.strip()

        # Process and display the response incrementally
        chunk_size = 1000
        chunk_accumulated = ""
        response_buffer += final_answer

        # Filtering and chunking logic
        if len(chunk_accumulated) >= chunk_size or "Query:" in response_buffer or "Original Answer:" in response_buffer or '.' in response_buffer or "Rewrite:" in response_buffer:
            if response_buffer.count("Original Answer:") > 1:
                response_buffer = remove_original_answers(response_buffer)
            response_buffer = filter_query(response_buffer)
            full_response += response_buffer
            response_buffer = ""
            st.session_state['results'].markdown(full_response)

        # Update session state with the new interaction
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append({'role': 'user', 'content': message})
        st.session_state['history'].append({'role': 'assistant', 'content': full_response})


# Function to filter out everything between "Query:" and "Original Answer:"
def filter_query(response_text):
    filtered_text = re.sub(r"Query:.*?(Original Answer:|Rewrite: |\?)", "", response_text, flags=re.DOTALL)
    filtered_text = filtered_text.replace("Original Answer:", "").replace("Rewrite:", "").replace("?", "")
    return filtered_text

# Function to handle multiple occurrences of 'Original Answer:'
def remove_original_answers(response_text):
    parts = response_text.split("Original Answer:")
    if len(parts) > 1:
        response_text = "Original Answer:" + parts[-1]  # Keep only the last 'Original Answer:'
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