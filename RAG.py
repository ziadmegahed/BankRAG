import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_groq import ChatGroq
import warnings

# Suppress Pydantic warnings
#warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ✅ Initialize Groq LLM with correct parameters
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0.1,
    max_tokens=1024,  # Direct parameter, not in model_kwargs
    model_kwargs={
        'top_p': 0.9,
        'frequency_penalty': 0.1,
    }
)

# ✅ Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# ✅ Load PDF documents
loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
documents = loader.load()

# ✅ Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# ✅ Create or update vector store
vector_store = Chroma.from_documents(
    texts,
    embeddings,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="stores/Banking_cosine"
)

print("✅ Vector Store Created Successfully!")