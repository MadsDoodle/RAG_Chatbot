import os
import uuid
import pickle
from typing import List
import glob

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from base64 import b64decode

from unstructured.partition.pdf import partition_pdf

# --- CONFIGURATION ---
# IMPORTANT: For deployment on Render, use a persistent disk path.
# We assume the disk is mounted at /data/
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "/data/multi_modal_rag_chroma_db")
DOCSTORE_PATH = os.environ.get("DOCSTORE_PATH", "/data/multi_modal_rag_docstore.pkl")
SOURCE_DOCS_PATH = "source_documents/6._price_trends.pdf"


# Ensure parent directories exist
os.makedirs(os.path.dirname(CHROMA_DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DOCSTORE_PATH), exist_ok=True)


# --- GLOBAL VARIABLES & MODELS ---
embedding_function = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
id_key = "doc_id"

# --- HELPER FUNCTIONS ---
def get_images_base64(chunks: List) -> List[str]:
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

# --- CORE LOGIC ---
def process_pdf(file_path: str):
    """
    Parses a PDF, creates summaries for text, tables, and images,
    and populates a vector store and document store.
    """
    print(f"Starting processing for PDF: {file_path}")

    # 1. Partition PDF into chunks
    raw_pdf_elements = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"], # We will handle tables as text for better summarization
    )

    # 2. Categorize elements
    texts, tables, images = [], [], []
    for el in raw_pdf_elements:
        el_type = type(el).__name__
        if el_type == "Table":
            # Treat tables as structured text for better LLM understanding
            texts.append(str(el))
        elif el_type in ["NarrativeText", "Title", "ListItem", "FigureCaption"]:
            texts.append(str(el))
        elif el_type == "Image":
            images.append(el.metadata.image_base64)
    
    print(f"Found {len(texts)} text/table chunks and {len(images)} images.")

    # 3. Create Summaries
    text_summarizer_prompt = ChatPromptTemplate.from_template(
        """You are an analyst summarizing text and tables. Provide a concise, bulleted summary of the key information. Element: {element}"""
    )
    summarize_text_chain = {"element": lambda x: x} | text_summarizer_prompt | llm | StrOutputParser()
    text_summaries = summarize_text_chain.batch(texts, {"max_concurrency": 5})

    image_summarizer_prompt_template = """
    You are an expert economic data analyst. Analyze the provided chart from a report on Price Trends. Create a structured summary.
    - **Identify Chart Type:** (e.g., Line Chart, Bar Chart).
    - **Extract Title:** The exact title.
    - **Describe Axes:** What the Y-axis and X-axis represent.
    - **Summarize Main Trend:** The key insight the chart shows.
    - **Extract Key Data Points:** List 2-3 specific data points.
    Respond in a bulleted list.
    """
    image_prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {"type": "text", "text": image_summarizer_prompt_template},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
        ])
    ])
    summarize_image_chain = image_prompt | llm | StrOutputParser()
    image_summaries = summarize_image_chain.batch([{"image": img} for img in images], {"max_concurrency": 5})
    
    print("Finished generating summaries.")

    # 4. Populate Stores
    vectorstore = Chroma(
        collection_name="rag_collection",
        embedding_function=embedding_function,
        persist_directory=CHROMA_DB_PATH
    )
    docstore = InMemoryStore()

    # Add text summaries
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_docs = [Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)]
    vectorstore.add_documents(summary_docs)
    docstore.mset(list(zip(doc_ids, texts)))

    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img_docs = [Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)]
    vectorstore.add_documents(summary_img_docs)
    docstore.mset(list(zip(img_ids, images)))

    print(f"Finished processing {os.path.basename(file_path)}")

def build_database_if_needed():
    """Checks if the database exists. If not, builds it from source documents."""
    if os.path.exists(CHROMA_DB_PATH):
        print("Database already exists. Loading retriever.")
        return get_retriever()

    print("No existing database found. Building from source documents...")
    vectorstore = Chroma(collection_name="rag_collection", embedding_function=embedding_function, persist_directory=CHROMA_DB_PATH)
    docstore = InMemoryStore()

    # Find all PDF files in the source directory
    pdf_files = glob.glob(os.path.join(SOURCE_DOCS_PATH, "*.pdf"))
    if not pdf_files:
        print("⚠️ No PDF files found in the source_documents directory.")
        return None

    for pdf_file in pdf_files:
        process_pdf(pdf_file, vectorstore, docstore)


    # 5. Persist to disk
    vectorstore.persist()
    with open(DOCSTORE_PATH, "wb") as f:
        pickle.dump(docstore, f)
    
    print("Successfully processed and stored PDF data.")

def get_retriever():
    """Loads the retriever from persisted storage."""
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_function,
            collection_name="rag_collection"
        )
        with open(DOCSTORE_PATH, "rb") as f:
            store = pickle.load(f)

        return MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)
    except Exception as e:
        print(f"Error loading retriever: {e}")
        return None

# --- RAG CHAIN ---
def create_rag_chain(retriever):
    """Creates the main RAG chain for answering questions."""

    def parse_retrieved_docs(docs):
        """Separates retrieved documents into text and image lists."""
        texts, images = [], []
        for doc in docs:
            if isinstance(doc, str) and len(doc) > 1000: # Heuristic for base64
                try:
                    b64decode(doc, validate=True)
                    images.append(doc)
                except:
                    texts.append(doc)
            elif isinstance(doc, str):
                 texts.append(doc)
            elif isinstance(doc, Document):
                texts.append(doc.page_content)
        return {"images": images, "texts": texts}

    def build_final_prompt(kwargs):
        """Constructs the final multi-modal prompt for the LLM."""
        context = kwargs["context"]
        question = kwargs["question"]
        
        context_text = "\n\n---\n\n".join(context["texts"])
        
        system_prompt = """
        You are an expert economic analyst AI. Answer the user's question based ONLY on the following context, which can include text, tables, and images.
        If the information is not in the context, say "I could not find information on that topic in the provided documents."
        """
        
        prompt_content = [{"type": "text", "text": f"{system_prompt}\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{question}"}]

        for image in context["images"]:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image.strip()}"},
            })
        
        return [HumanMessage(content=prompt_content)]

    chain = (
        {
            "context": retriever | RunnableLambda(parse_retrieved_docs),
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough.assign(
            response=(
                RunnableLambda(build_final_prompt)
                | llm
                | StrOutputParser()
            )
        )
    )
    return chain