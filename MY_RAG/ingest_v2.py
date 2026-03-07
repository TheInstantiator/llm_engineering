
import os
import threading
import glob
import json
import hashlib
import concurrent.futures
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 0. Configuration & Environment Setup
# ----------------------------------
# load_dotenv reads the '.env' file and adds its variables to the 'os.environ' dictionary.
# In Java, you might use a .properties or .yaml file for this.
load_dotenv(override=True)

# Map GROK_API_KEY to XAI_API_KEY for LiteLLM. 
# LiteLLM is a library that allows you to call multiple LLM providers (OpenAI, Anthropic, xAI) 
# using the exact same standard format.
if os.getenv("GROK_API_KEY") and not os.getenv("XAI_API_KEY"):
    os.environ["XAI_API_KEY"] = os.getenv("GROK_API_KEY")

import chromadb
# We use sentence_transformers purely for the local embeddings (Native Python, NO LangChain)
from sentence_transformers import SentenceTransformer

# LiteLLM for the "AI Agent" processing steps (Summarization, Pre-screening)
import litellm
import pymupdf4llm
from tqdm import tqdm

# Disable telemetry / tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
litellm.telemetry = False

# ==========================================
# 1. Pydantic Models (The "Native" Schema)
# ==========================================
# Pydantic is much more than a simple class. Think of it like a Java Interface + Validator.
# It ensures that data entering your script matches the expected types (str, int, etc).
# If you try to pass an integer to 'file_path', Pydantic will throw an error immediately.

class ChunkMetadata(BaseModel):
    """Metadata for each text chunk. Used for filtering in RAG later."""
    file_path: str
    doc_hash: str
    doc_summary: str
    parent_heading: str
    chunk_index: int

class RAGChunk(BaseModel):
    """The core container for our data. 'Field' is used to add descriptions/rules."""
    id: str = Field(..., description="Unique ID: hash of file_path + chunk_index")
    content: str
    metadata: ChunkMetadata

class DocumentReview(BaseModel):
    """
    Structured output expected from the LLM. 
    By passing this model to LiteLLM, we force the AI to return JSON that 
    matches this EXACT schema—no more messy text parsing!
    """
    semantic_summary: str = Field(description="A concise summary of the entire document's core meaning.")
    is_technical: bool = Field(description="Does this document contain heavy technical data/tables?")

# ==========================================
# 2. The Native Ingestion Agent System
# ==========================================

class IngestionAgentPipeline:
    def __init__(self, config_path: str = "config.json"):
        # Threads in Python share memory. To prevent Two threads from writing to the 
        # same shared variable (like total_cost) at the same time, we use a 'Lock'.
        # This is exactly like 'synchronized' blocks in Java.
        self._embedding_lock = threading.Lock()
        self._cost_lock = threading.Lock()
        self.total_cost = 0.0
        self.total_tokens = 0
        
        # Load Config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.db_path = self.config.get("db_path", "./chroma_db_v2")
        self.llm_model = self.config.get("llm_model", "gpt-4o-mini")
        
        # Initialize Local Native Embedding Encoder
        encoder_model = self.config.get("embedding_model", "Alibaba-NLP/gte-Qwen2-7B-instruct")
        print(f"Loading Local Native Encoder: {encoder_model}...")
        
        # Determine Device
        import torch
        # By default, use CPU to avoid OOM for 7B models on 8GB cards like in the original script.
        # User must explicitly set RAG_DEVICE=cuda to use GPU.
        env_device = os.getenv("RAG_DEVICE", "cpu")
        device = env_device
        if env_device == "cuda" and not torch.cuda.is_available():
            print("  --> WARNING: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
            
        print(f"  --> Processing on: {device.upper()}")
        
        self.encoder = SentenceTransformer(encoder_model, device=device, trust_remote_code=True)
        print("  --> Local Encoder loaded into memory successfully!")
        
        # Parallel Execution Config
        self.max_workers = self.config.get("max_workers", 4)
        print(f"  --> Parallel Workers Limit: {self.max_workers}")
        
        # Initialize Native ChromaDB Client
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="wms_docs_v2",
            metadata={"hnsw:space": "cosine"}
        )

    def calculate_md5(self, file_path: str) -> Optional[str]:
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None

    def _agentic_review_document(self, content: str) -> DocumentReview:
        """
        Uses LiteLLM to act as an Agent, reviewing the raw document and returning structured Pydantic data.
        This provides the semantic summary that gets embedded into every chunk's metadata!
        """
        # We sample the first 4000 chars to avoid massive token costs during summary
        sample_text = content[:4000]
        
        try:
            # LiteLLM allows us to pass Pydantic models directly to any provider (OpenAI, Gemini, Anthropic)
            # that supports structured outputs!
            kwargs = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": "You are a senior technical screener. Summarize the core intent of the following document excerpt."},
                    {"role": "user", "content": sample_text}
                ],
                "response_format": DocumentReview,
                "temperature": 0.1,
                "timeout": 15
            }
            
            # If using Grok/xAI, explicitly pass their specific API base URL to prevent routing hangs
            if "xai" in self.llm_model.lower() or "grok" in self.llm_model.lower():
                kwargs["api_base"] = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
                
            print(f"    [Agent] Requesting summary from {self.llm_model}...")
            response = litellm.completion(**kwargs)
            
            # Extract and log token usage/costs
            try:
                cost = litellm.completion_cost(completion_response=response)
                tokens = response.usage.total_tokens
                with self._cost_lock:
                    if cost:
                        self.total_cost += cost
                    if tokens:
                        self.total_tokens += tokens
            except Exception as metric_e:
                pass # Silent fail if unsupported model doesn't return cost data
            
            # Parse the JSON string returned by the provider into our Pydantic model
            raw_response = response.choices[0].message.content
            print(f"    [Agent] Success! Summary generated: {raw_response[:75]}...")
            return DocumentReview.model_validate_json(raw_response)
        except Exception as e:
            print(f"\n  ---> ❌ [Agent Error] Model {self.llm_model} failed to review document.")
            print(f"  ---> ❌ [Error Details]: {type(e).__name__}: {str(e)}\n")
            return DocumentReview(semantic_summary="Summary unavailable.", is_technical=False)

    def _chunk_text_semantic(self, text: str, file_path: str, summary: str, doc_hash: str) -> List[RAGChunk]:
        """A smarter chunker using LangChain."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        chunk_size = self.config["chunk_settings"]["chunk_size"]
        overlap = self.config["chunk_settings"]["chunk_overlap"]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        text_chunks = splitter.split_text(text)
        
        chunks = []
        for idx, chunk_content in enumerate(text_chunks):
            chunk_id = hashlib.md5(f"{file_path}_{idx}".encode()).hexdigest()
            metadata = ChunkMetadata(
                file_path=file_path,
                doc_hash=doc_hash,
                doc_summary=summary,
                parent_heading="General",
                chunk_index=idx
            )
            chunks.append(RAGChunk(
                id=chunk_id,
                content=chunk_content,
                metadata=metadata
            ))
            
        return chunks

    def _process_single_file(self, file_path: str):
        """Helper method to process a single file, designed to run in a thread."""
        current_hash = self.calculate_md5(file_path)
        if current_hash:
            try:
                existing_docs = self.collection.get(
                    where={"doc_hash": current_hash},
                    limit=1
                )
                if existing_docs and existing_docs.get('ids') and len(existing_docs['ids']) > 0:
                    print(f"  --> Skipping [Unchanged]: {os.path.basename(file_path)}")
                    return
            except Exception:
                pass # Database or collection might be empty/new

        print(f"  --> Processing: {file_path}")
        try:
            if file_path.lower().endswith(".pdf"):
                content = pymupdf4llm.to_markdown(file_path)
            elif file_path.lower().endswith(".xlsx"):
                import pandas as pd
                # Read all sheets from the Excel file
                all_sheets = pd.read_excel(file_path, sheet_name=None)
                md_content = []
                for sheet_name, df in all_sheets.items():
                    # Strip out entirely empty rows/columns caused by stray Excel formatting
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    
                    # Only append if the sheet actually had data
                    if not df.empty:
                        md = df.to_markdown(index=False)
                        
                        # Soft limit: If a single tab produces an enormous markdown string (like Gantt charts), skip it!
                        if len(md) > 100000:
                            print(f"  --> ⚠️ WARNING: Sheet '{sheet_name}' in {os.path.basename(file_path)} is too massive ({len(md)} chars). Skipping this specific tab.")
                            continue
                            
                        md_content.append(f"## Sheet: {sheet_name}\n")
                        md_content.append(md)
                        md_content.append("\n\n")
                content = "".join(md_content)
            elif file_path.lower().endswith(".docx"):
                import docx
                doc = docx.Document(file_path)
                content = "\n".join([p.text for p in doc.paragraphs])
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
        except Exception as e:
            print(f"  --> Failed to read {os.path.basename(file_path)}: {e}")
            return
            
        if not content.strip():
            return
            
        # 1. Agent pre-screens the document to generate a semantic summary
        doc_review = self._agentic_review_document(content)
        
        # 2. Chunk the text, passing the Agent's summary and the file hash into the metadata
        chunks = self._chunk_text_semantic(content, file_path, doc_review.semantic_summary, current_hash or "unknown")
        
        if not chunks:
            return
            
        if len(chunks) > 500:
            print(f"  --> ⚠️ WARNING: {os.path.basename(file_path)} generated {len(chunks)} chunks. Skipping because file is abnormally large.")
            return
            
        # 3. Embed all chunks (Native Python execution)
        # We turn the text into numbers (vectors) so the computer can 'understand' it.
        texts_to_embed = [c.content for c in chunks]
        
        # We lock this section because the local embedding models (using PyTorch) 
        # can sometimes deadlock or crash if multiple threads try to use them simultaneously 
        # on the same machine/CPU.
        with self._embedding_lock:
            print(f"    [Agent] Vectorizing {len(texts_to_embed)} chunks...", flush=True)
            embeddings = self.encoder.encode(texts_to_embed, normalize_embeddings=True)
            
            # 4. Insert directly into ChromaDB using Chroma's native APIs
            # 'model_dump()' converts our Pydantic objects into standard Python dictionaries.
            self.collection.add(
                ids=[c.id for c in chunks],
                documents=[c.content for c in chunks],
                embeddings=embeddings.tolist(),
                metadatas=[c.metadata.model_dump() for c in chunks]
            )
            print(f"    [Agent] Successfully saved chunks for {os.path.basename(file_path)}!", flush=True)

    def process_folder(self, folder_path: str):
        print(f"\n🚀 Agent Scanning Folder: {folder_path}")
        
        target_files = []
        target_files += glob.glob(os.path.join(folder_path, "**", "*.md"), recursive=True)
        target_files += glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True)
        target_files += glob.glob(os.path.join(folder_path, "**", "*.pdf"), recursive=True)
        target_files += glob.glob(os.path.join(folder_path, "**", "*.xlsx"), recursive=True)
        target_files += glob.glob(os.path.join(folder_path, "**", "*.docx"), recursive=True)
        
        # Exclude 'desktop' files
        target_files = [f for f in target_files if "desktop" not in os.path.basename(f).lower()]
        
        # Exclude '.zip' and explicitly excluded folders
        excluded_folders = self.config.get("excluded_folders", [])
        filtered_files = []
        for f in target_files:
            if ".zip" in f.lower():
                continue
                
            skip = False
            for ex_dir in excluded_folders:
                if f.startswith(ex_dir):
                    skip = True
                    break
            if not skip:
                filtered_files.append(f)
                
        target_files = filtered_files
        
        if not target_files:
            print("  --> No supported files found in this directory.")
            return

        # Parallelize the file processing loop
        print(f"  --> Initiating thread pool with {self.max_workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # We use list(tqdm(executor.map(...))) to evaluate the generator and show progress
            list(tqdm(
                executor.map(self._process_single_file, target_files),
                total=len(target_files),
                desc="Processing Files (Parallel)"
            ))

if __name__ == "__main__":
    print("--- NATIVE RAG AGENT PIPELINE (LiteLLM + ChromaDB + Pydantic) ---")
    
    # Safely resolve the config path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    pipeline = IngestionAgentPipeline(config_path)
    
    folders = pipeline.config.get("included_folders", [])
    for folder in folders:
        pipeline.process_folder(folder)
        
    print(f"\n✅ Finished! Collection now contains {pipeline.collection.count()} chunks.")
    print(f"💰 Total LLM API Cost: ${pipeline.total_cost:.6f} ({pipeline.total_tokens} tokens used)")
