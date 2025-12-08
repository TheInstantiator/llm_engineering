# We use the OpenAI library to access embedding models and potentially other API features.
from openai import OpenAI
# load_dotenv allows us to load environment variables (like API keys) from a .env file.
from dotenv import load_dotenv
# PersistentClient from chromadb is used to interact with the vector database where we store our documents.
from chromadb import PersistentClient
# completion from litellm is a unified interface to call various LLM APIs (OpenAI, Anthropic, Groq, etc.) interchangeably.
from litellm import completion
# Pydantic is used for data validation and defining the structure of our data models.
from pydantic import BaseModel, Field
from pathlib import Path
# Tenacity is a library to handle retrying operations that might fail transiently (like network requests).
from tenacity import retry, wait_exponential


# Load environment variables from .env file, overriding any existing system variables.
load_dotenv(override=True)

# MODEL = "openai/gpt-4.1-nano"
# We are using a specific model hosted on Groq. This string format is specific to LiteLLM.
# LiteLLM is a lightweight library that provides a unified interface for calling various LLM APIs
# (like OpenAI, Anthropic, Azure, Groq, etc.) using the standard OpenAI format.
# It handles the translation of inputs and outputs, allowing you to switch between different
# model providers simply by changing this model string, without rewriting your code.
# The format usually follows 'provider/model-name'.
MODEL = "groq/openai/gpt-oss-120b"
# Paths to our database and knowledge base directories.
# We use Path(__file__).parent to locate these relative to this script's location.
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
SUMMARIES_PATH = Path(__file__).parent.parent / "summaries"

collection_name = "docs"
# The embedding model used to convert text into vector representations.
# "text-embedding-3-large" is a specific model provided by OpenAI known for high performance.
# This must match the model used when the database was populated.
embedding_model = "text-embedding-3-large"
# Configure a wait strategy for retries: exponential backoff starting at 10s, up to 240s.
wait = wait_exponential(multiplier=1, min=10, max=240)

# Initialize the OpenAI client for embedding generation.
openai = OpenAI()

# Connect to the persistent Chroma vector database at the specified path.
chroma = PersistentClient(path=DB_NAME)
# Get or create the collection where our document embeddings are stored.
collection = chroma.get_or_create_collection(collection_name)

# Number of documents to retrieve initially from the vector database.
# 20 is a common widely-used starting point (often between 10-50). Factors for determining this include:
# - Recall vs. Latency/Cost: A higher K increases the chance of capturing relevant info (recall) but requires more processing time and cost for the reranker.
# - Reranker Limitation: Reranking is computationally expensive; you typically retrieve a broader set here to filter down later.
RETRIEVAL_K = 20
# Number of documents to keep after reranking.
# 10 is a balanced setting. Factors include:
# - Context Window: You must ensure the combined text of these chunks fits within the LLM's token limit.
# - "Lost in the Middle" Phenomenon: Providing too much context can confuse models; they often focus on the beginning and end.
# - Cost: More context equals more input tokens, which increases API costs.
FINAL_K = 10

# The system prompt defines the AI's persona and instructions for the final answer generation.
# It includes a placeholder {context} where the retrieved information will be inserted.
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


# Define a data model for a search result, containing the text content and metadata.
class Result(BaseModel):
    page_content: str
    metadata: dict


# Define a model for the reranking output. The LLM will return a list of indices indicating the order of relevance.
class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


# Reranking function to re-order the retrieved chunks based on their relevance to the question.
# This uses an LLM to evaluate relevance, which is often more accurate than simple vector similarity.
# We use the @retry decorator to handle potential API failures automatically.
@retry(wait=wait)
def rerank(question, chunks):
    # The system prompt for the reranker explains its specific task: to order chunks by relevance.
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    # The user prompt includes the question and the chunks to be ranked.
    # We formatting the prompt to clearly present the chunks.
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Call the LLM with structured output (response_format=RankOrder) to get a predictable JSON response.
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]


# Clear logic to construct the message history for the final RAG generation.
# It combines the system prompt (with context), conversation history, and the new user question.
def make_rag_messages(question, history, chunks):
    # Format the retrieved chunks into a single string to inject into the system prompt.
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


# Query rewriting function.
# It uses the conversation history and the current question to formulate a better search query for the database.
# This helps when the user asks vague follow-up questions like "Tell me more about that".
@retry(wait=wait)
def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


# Helper function to combine two lists of chunks (e.g., from original and rewritten queries) without duplicates.
def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


# Perform the initial vector search in ChromaDB.
# This converts the question to an embedding and finds the nearest neighbors.
def fetch_context_unranked(question):
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks


# Orchestrates the full retrieval process:
# 1. Retrieve based on original question.
# 2. Rewrite question and retrieve based on that.
# 3. Merge results.
# 4. Rerank the merged results using the LLM.
def fetch_context(original_question):
    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]


# Main function to answer a user's question.
# It coordinates the context fetching, message construction, and final answer generation.
@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks

# ==========================================
# SUPERVISOR SUGGESTIONS
# ==========================================
# 1. Configuration: Move model names and parameters (like K values) to a configuration file or environment variables for easier adjustment without code changes.
# 2. Asynchronous Support: Calls to OpenAI and ChromaDB differ in latency. Using async/await (e.g., AsyncOpenAI, async completion) could improve throughput for a web server context.
# 3. Error Handling: While 'retry' handles transient errors, explicit try/except blocks around external API calls would provide better fallback or user feedback mechanisms.
# 4. Type Hinting: Expand type hints (e.g., using 'List', 'Dict', 'Optional' from typing) to all functions for better code safety and IDE support.
# 5. Logging: Replace print statements (if any) or implicit behaviors with a proper logging setup (import logging) to track RAG performance and errors.
