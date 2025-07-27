import os
from dotenv import load_dotenv
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

MODEL = "gpt-4o-mini"
DB_NAME = "vector_db"

# Load documents, create vectorstore and return conversation chain
def load_docs_and_create_chain(directory_path):
    all_docs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory_path, filename))
            documents = loader.load()
            all_docs.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()

    # Clear previous vector DB if exists
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(documents=chunked_docs, embedding=embeddings, persist_directory=DB_NAME)

    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print(retriever)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return conversation_chain

# Setup function
def setup_chat(dir_path):
    try:
        chain = load_docs_and_create_chain(dir_path)
        return gr.update(visible=True), "", chain
    except Exception as e:
        return gr.update(visible=False), f"Error: {str(e)}", None

# Message handler — using OpenAI-style messages
def user_chat(message, history, chain):
    if chain is None:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Chat is not ready. Please load documents first."}
        ]
    result = chain.invoke({"question": message})
    return history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result["answer"]}
    ]

# Gradio app
with gr.Blocks() as demo:
    dir_input = gr.Textbox(label="PDF Document Directory path", value="C:\Ofc_Docs\Abinitio\Abinitio_dml_functions")
    submit_btn = gr.Button("Load Documents")
    status_box = gr.Markdown("🔄 Loading...", visible=False)
    error_output = gr.Textbox(label="Error", visible=False)
    chain_state = gr.State()

    with gr.Group(visible=False) as chat_container:
        chatbot = gr.Chatbot(label="Chat with Your Documents", type="messages")
        msg = gr.Textbox(placeholder="Ask a question...", label="Message")
        send_btn = gr.Button("Send")

    submit_btn.click(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[status_box]
    ).then(
        fn=setup_chat,
        inputs=[dir_input],
        outputs=[chat_container, error_output, chain_state]
    ).then(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[status_box]
    )

    # Click to send
    send_btn.click(
        fn=user_chat,
        inputs=[msg, chatbot, chain_state],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        inputs=[],
        outputs=[msg]
    )

    # Press Enter to send
    msg.submit(
        fn=user_chat,
        inputs=[msg, chatbot, chain_state],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        inputs=[],
        outputs=[msg]
    )

demo.launch(share=True)
