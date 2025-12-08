# pathlib is used for robust and cross-platform file path manipulations.
from pathlib import Path
# OpenAI client is used here specifically for generating embeddings.
from openai import OpenAI
# load_dotenv loads environment variables from a .env file, crucial for keeping API keys secure.
from dotenv import load_dotenv
# Pydantic is used to define strict data schemas (models) for our chunks and results, ensuring data consistency.
from pydantic import BaseModel, Field
# chromadb.PersistentClient allows us to save our vector database to disk so it persists between runs.
from chromadb import PersistentClient
# tqdm is a library that provides a progress bar for our loops, very helpful for long-running ingestion processes.
from tqdm import tqdm
# completion from litellm provides a unified interface to call various LLM APIs (OpenAI, Anthropic, etc.).
from litellm import completion
# multiprocessing.Pool allows us to parallelize the document processing, significantly speeding up ingestion.
from multiprocessing import Pool
# tenacity is used for retrying operations (like API calls) that might fail transiently.
from tenacity import retry, wait_exponential


# Load environment variables.
load_dotenv(override=True)

# The model used for generating valid chunks from raw text.
# Changing this to a more capable model (chunking is a complex task) can improve retrieval quality.
MODEL = "openai/gpt-4.1-nano"

# Define file paths relative to this script.
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"
# The embedding model used to convert text chunks into vectors.
# "text-embedding-3-large" is a high-performance model from OpenAI.
embedding_model = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
# A heuristic for estimation; used to guide the LLM on how many chunks to create.
AVERAGE_CHUNK_SIZE = 100
# Retry configuration: wait exponentially between retries, tailored for API rate limits.
wait = wait_exponential(multiplier=1, min=10, max=240)


WORKERS = 3

openai = OpenAI()


# Result model represents the final structure derived from a chunk, ready for indexing or use.
class Result(BaseModel):
    page_content: str
    metadata: dict


# Chunk model defines the structured output we expect from the LLM when it splits the document.
# Instead of naive splitting (by character count), we ask the LLM to semantically split and summarize.
class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    # Helper method to convert the LLM-generated Chunk into a generic Result format.
    # It combines headline, summary, and original text into the final searchable content.
    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


# Container model for a list of chunks, used for structured output parsing.
class Chunks(BaseModel):
    chunks: list[Chunk]


# Function to iterate over the knowledge base directory and load all markdown files.
# It returns a list of dictionaries with document metadata and content.
def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""

    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        # Recursively find all .md files.
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents


# Constructs the prompt for the LLM to perform "agentic chunking".
# Instead of fixed-size chunks, we instruct the LLM to divide the text logically and provide metadata (headline, summary).
def make_prompt(document):
    # Calculate a rough estimate of expected chunks to guide the LLM.
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]


# Worker function to process a single document.
# It calls the LLM to split the document into chunks and then formats them.
# The @retry decorator handles potential API failures.
@retry(wait=wait)
def process_document(document):
    messages = make_messages(document)
    # We enforce the 'Chunks' structure using 'response_format' (JSON mode).
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


# Orchestrates the parallel processing of all documents.
# It uses a process pool to run 'process_document' on multiple cores concurrently.
def create_chunks(documents):
    """
    Create chunks using a number of workers in parallel.
    If you get a rate limit error, set the WORKERS to 1.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        # imap_unordered is used for better efficiency when order doesn't matter.
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


# Takes the processed text chunks, generates embeddings, and saves them to the ChromaDB.
def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    # Reset collection by deleting if it exists - be careful with this in production!
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    # Generate embeddings for all chunks in one batch (or fewer batches internally).
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(collection_name)

    # Generate simple IDs and add data to the collection.
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")

# ==========================================
# SUPERVISOR SUGGESTIONS
# ==========================================
# 1. Chunking Strategy: Agentic chunking (using an LLM to split text) is high-quality but slow and expensive. Consider hybrid approaches or faster models if scaling up.
# 2. Embedding Batching: Sending all texts to the embedding API at once (line 169) might hit payload limits. Batching these requests (e.g., groups of 100) is safer.
# 3. Incremental Updates: Currently, the script deletes and recreates the collection every time (line 165). For production, implement logic to only add new/modified documents.
# 4. Error Handling: Add robust error handling in the worker process (process_document) to ensure one bad document doesn't crash the whole batch (though Pool handles some of this).
# 5. Rate Limiting: When using multiple workers, you might hit OpenAI's rate limits faster. Implementing a token bucket or using 'tenacity' more aggressively might be needed.
