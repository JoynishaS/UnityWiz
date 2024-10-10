import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex,StorageContext, ServiceContext,load_index_from_storage,GPTVectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

#Configure settings for the application
Settings.text_splitter = SentenceSplitter(chunk_size=1000,chunk_overlap=20)
Settings.embed_model = NVIDIAEmbedding(model = "NV-Embed-QA", truncate="END", api_key= st.secrets['NVIDIA_API_KEY'] )

# Load your model and tokenizer
model_path = "./llama-finetuned-test-unity"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16)

# Make sure the model is in evaluation mode
model.eval()

# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Use torch.compile to optimize the model
compiled_model = torch.compile(model)

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

        # Use the tokenizer separately to prepare inputs with truncation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            # Generate text
            output_ids = compiled_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,  # Controls total output length
                #max_length=inputs["input_ids"].shape[1] + 100,  # Limit max_length accordingly
                num_return_sequences=1,
                pad_token_id=model.config.pad_token_id,
                do_sample=True,
                top_k=30,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.2,  # Discourages exact repetition
            )
        # Decode and print the generated text, excluding the prompt from the result
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

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