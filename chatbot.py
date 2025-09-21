# Install packages if not already installed:
# pip install ollama langchain langchain-community chromadb sentence-transformers gradio pypdf

import ollama
from langchain.llms.base import LLM
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr
import os

# ------------------------------
# LangChain wrapper for Ollama
# ------------------------------
class OllamaLLM(LLM):
    @property
    def _llm_type(self):
        return "ollama"

    def _call(self, prompt, stop=None):
        response = ollama.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.8}
        )
        return response["message"]["content"]

llm = OllamaLLM()

# ------------------------------
# Load PDFs and split text
# ------------------------------
pdf_dir = "data/"
if not os.path.exists(pdf_dir):
    raise FileNotFoundError(f"Directory not found: '{pdf_dir}'")

loader = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)

# ------------------------------
# Create embeddings + vector DB
# ------------------------------
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(texts, embeddings, persist_directory="content/chromadb")
vectordb.persist()

# ------------------------------
# Prompt template for empathy
# ------------------------------
prompt_template = """
You are a kind, empathetic mental health assistant for youth.
Validate feelings, suggest healthy coping strategies (breathing, journaling, music, exercise).
Never give medical diagnoses.
Context: {context}
User question: {question}
Chatbot:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ------------------------------
# Setup RetrievalQA
# ------------------------------
qachain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT}
)

# ------------------------------
# Crisis detection
# ------------------------------
CRISIS_KEYWORDS = ["suicide", "kill myself", "self-harm", "end my life"]
HOTLINE_MESSAGE = (
    "I'm concerned about your safety. Please contact a professional immediately.\n"
    "- India: 022 2754 6669 (Vandrevala Foundation)\n"
    "- International: +1 800 273 8255 (US)"
)

# ------------------------------
# Gradio chatbot function
# ------------------------------
def chatbot_response(user_input, history):
    if not user_input.strip():
        return "Please provide a valid input.", history
    
    if any(word in user_input.lower() for word in CRISIS_KEYWORDS):
        response = HOTLINE_MESSAGE
    else:
        response = qachain.run(user_input)
    
    history.append((user_input, response))
    return "", history

# ------------------------------
# Gradio UI
# ------------------------------
with gr.Blocks() as app:
    gr.Markdown("# Youth Mental Health Chatbot")
    gr.Markdown("A compassionate chatbot for youth mental wellness. *Note:* For urgent concerns, please contact professionals directly.")
    
    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(label="Your message")
    submit = gr.Button("Send")
    
    submit.click(chatbot_response, inputs=[msg, chatbot_ui], outputs=[msg, chatbot_ui])

if _name_== "__name__":
    app.launch()