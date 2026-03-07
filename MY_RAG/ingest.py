# /// script
# dependencies = [
#     "langchain-community",
#     "langchain-text-splitters",
#     "langchain-huggingface",
#     "langchain-google-genai",
#     "langchain-chroma",
#     "chromadb",
#     "sentence-transformers",
#     "python-dotenv",
#     "langchain",
#     "langchain-core",
#     "einops",
#     "transformers==4.46.3",
#     "accelerate>=0.26.0",
# ]
# ///

import os
import glob
import json
import hashlib
import time
from typing import List, Dict, Set, Tuple, Optional, Any

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load Env for LLM
load_dotenv(override=True)

class IngestionSystem:
    def __init__(self, source_path: str, db_path: str, model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"):
        self.source_path = source_path
        self.db_path = db_path
        self.state_file = os.path.join(db_path, "ingestion_state.json")
        self.model_name = model_name
        self.extensions = {'.md', '.txt', '.csv', '.py', '.json', '.html'}
        
        # Initialize lazily
        self._embeddings = None
        self._vectorstore = None
        self._text_splitter = None
        # self._llm = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            print(f"Loading Local Embeddings: {self.model_name}...")
            # Use GPU if available AND requested (Safe default for 7B models on 8GB cards)
            import torch
            # For 7B models, we need ~16GB VRAM. If on 8GB card, force CPU.
            # You can override this by setting RAG_DEVICE=cuda
            env_device = os.getenv("RAG_DEVICE", "cpu")
            device = env_device
            
            if env_device == "cuda" and not torch.cuda.is_available():
                print("  --> WARNING: CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
                
            print(f"  --> Running on device: {device.upper()}")
            
            # Using HuggingFace (Local)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"trust_remote_code": True, "device": device},
                encode_kwargs={"normalize_embeddings": True}
            )
        return self._embeddings

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            print(f"Loading Vector Database from: {self.db_path}...")
            self._vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
        return self._vectorstore

    @property
    def text_splitter(self):
        if self._text_splitter is None:
            # Using your notebook settings: 1000 char chunks, 250 overlap
            self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        return self._text_splitter

#     @property
#     def llm(self):
#         """LLM removed for speed optimization."""
#         pass

    def calculate_md5(self, file_path: str) -> Optional[str]:
        """Reads a file and returns its MD5 hash."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Could not hash {file_path}: {e}")
            return None

    def load_state(self) -> Dict[str, Dict[str, str]]:
        """Loads the ledger from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    # Migration: Convert old format (str) to new format (dict) if valid
                    # If state is empty or weird, just return it, valid checks run later
                    if state and isinstance(list(state.values())[0], str):
                        print("Migrating internal state to include summaries...")
                        return {k: {"hash": v, "summary": ""} for k, v in state.items()}
                    return state
            except json.JSONDecodeError:
                print("Warning: State file corrupted, starting fresh.")
                return {}
        return {}

    def save_state(self, state: Dict[str, Dict[str, str]]):
        """Saves the ledger to disk."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

#     def generate_summary(self, content: str, filename: str) -> str:
#         """Summarization disabled."""
#         return ""

    def scan_files(self) -> Tuple[Dict[str, Any], List[Tuple[str, str]], List[str], int]:
        """Scans source and compares with ledger."""
        print("Loading previous state...")
        previous_state = self.load_state()
        current_state = {}
        files_to_process = []
        files_to_delete = []

        print(f"Scanning {self.source_path} for changes...")
        search_pattern = os.path.join(self.source_path, "**")
        all_files = glob.glob(search_pattern, recursive=True)
        target_files = [f for f in all_files if os.path.splitext(f)[1].lower() in self.extensions]
        
        for i, file_path in enumerate(target_files):
            if i % 1000 == 0 and i > 0:
                print(f"  Scanned {i} files...")

            file_hash = self.calculate_md5(file_path)
            if file_hash is None:
                continue
                
            # Default entry for current state
            # If it exists in previous state and matches hash, copy over the summary so we don't re-gen it
            existing_entry = previous_state.get(file_path)
            
            if existing_entry and existing_entry.get("hash") == file_hash:
                 # State Unchanged - Keep old summary
                 current_state[file_path] = existing_entry
            else:
                # State Changed or New - Will need processing
                # We initialize with empty summary, will fill during processing
                current_state[file_path] = {"hash": file_hash, "summary": ""}
                
                if not existing_entry:
                    files_to_process.append(("NEW", file_path))
                else:
                    files_to_process.append(("MODIFIED", file_path))

        # Identify Deleted
        for old_path in previous_state:
            if old_path not in current_state:
                files_to_delete.append(old_path)

        return current_state, files_to_process, files_to_delete, len(target_files)

    def run(self):
        """Main execution method."""
        current_state, files_to_process, files_to_delete, total_files = self.scan_files()

        print(f"\n--- SCAN REPORT ---")
        print(f"Total Files Found: {total_files}")
        print(f"Files to Add/Update: {len(files_to_process)}")
        print(f"Files to Delete:     {len(files_to_delete)}")
        
        if len(files_to_process) == 0 and len(files_to_delete) == 0:
            print("\nSystem is up to date. Nothing to do!")
            self.save_state(current_state)
            return

        db = self.vectorstore

        # -- A. HANDLE DELETIONS --
        if files_to_delete:
            print(f"\nProcessing {len(files_to_delete)} deletions...")
            for del_path in files_to_delete:
                print(f"  - Deleting vectors for: {os.path.basename(del_path)}")
                try:
                    db._collection.delete(where={"source": del_path})
                except Exception as e:
                    print(f"    Error deleting {del_path}: {e}")

        # -- B. HANDLE ADDITIONS/UPDATES --
        if files_to_process:
            print(f"\nProcessing {len(files_to_process)} additions/updates...")
            
            chunk_batch = []
            # Local models can handle larger batches, but let's keep it safe to prevent RAM spikes
            BATCH_LIMIT = 50 
            
            for i, (action, file_path) in enumerate(files_to_process):
                try:
                    if action == "MODIFIED":
                        db._collection.delete(where={"source": file_path})
                    
                    # Load Content
                    loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
                    docs = loader.load()
                    
                    if not docs:
                        continue
                        
                    content = docs[0].page_content
                    
                    # 1. Generate Summary (Skipped)
                    print(f"  [{i+1}/{len(files_to_process)}] Ingesting: {os.path.basename(file_path)}")
                    # For huge datasets, printing every file might spam. Let's print every 10 or large files options.
                    
                    # summary = self.generate_summary(content, os.path.basename(file_path))
                    current_state[file_path]["summary"] = "" # summary
                    
                    # 2. Split and Buffer chunks
                    chunks = self.text_splitter.split_documents(docs)
                    if chunks:
                        chunk_batch.extend(chunks)
                    
                    # 3. Process Batch (Strict Slicing)
                    # We use a while loop to drain the buffer in chunks of BATCH_LIMIT
                    while len(chunk_batch) >= BATCH_LIMIT:
                        current_batch = chunk_batch[:BATCH_LIMIT]
                        chunk_batch = chunk_batch[BATCH_LIMIT:] # Remove processed
                        
                        try:
                            # print(f"    --> Committing local batch of {len(current_batch)} chunks...")
                            db.add_documents(current_batch)
                        except Exception as e:
                            print(f"    Batch Error: {e}")
                            # If a batch fails, we might lose data here, but safer than crashing
                        
                        # Progress Marker
                        if (i + 1) % 10 == 0 and len(chunk_batch) < BATCH_LIMIT:
                             print(f"  Processed {i+1}/{len(files_to_process)} files...")

                except Exception as e:
                    print(f"  FAILED to process {file_path}: {e}")

            # 4. Flush remaining chunks
            if chunk_batch:
                print(f"    --> Committing final batch of {len(chunk_batch)} chunks...")
                try:
                    db.add_documents(chunk_batch)
                except Exception as e:
                    print(f"    Final Batch Error: {e}")

        # -- C. COMMIT STATE --
        print("\nSaving new state to ledger...")
        self.save_state(current_state)
        print("SYNC COMPLETE! Database is now identical to source.")


if __name__ == "__main__":
    SOURCE_PATH = os.getenv("RAG_SOURCE_PATH", "/mnt/e/WMS_selection")
    DB_PATH = os.getenv("RAG_DB_PATH", "/mnt/e/chroma_db_wms")
    
    print("--- WMS RAG INGESTION SYSTEM (Gemini Embeddings + AI Summaries) ---")
    print(f"Source: {SOURCE_PATH}")
    print(f"Database: {DB_PATH}")
    
    if not os.path.exists(SOURCE_PATH):
        print(f"ERROR: Source path '{SOURCE_PATH}' does not exist.")
        exit(1)
        
    system = IngestionSystem(source_path=SOURCE_PATH, db_path=DB_PATH)
    system.run()
