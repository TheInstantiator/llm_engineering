
# /// script
# dependencies = [
#     "langchain-community",
#     "langchain-text-splitters",
#     "langchain-huggingface",
#     "langchain-chroma",
#     "chromadb",
#     "sentence-transformers",
#     "langchain-openai",
#     "python-dotenv",
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
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load Env for LLM
load_dotenv(override=True)

class IngestionSystem:
    def __init__(self, source_path: str, db_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.source_path = source_path
        self.db_path = db_path
        self.state_file = os.path.join(db_path, "ingestion_state.json")
        self.model_name = model_name
        self.extensions = {'.md', '.txt', '.csv', '.py', '.json', '.html'}
        
        # Initialize lazily
        self._embeddings = None
        self._vectorstore = None
        self._text_splitter = None
        self._llm = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
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
            self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return self._text_splitter

    @property
    def llm(self):
        if self._llm is None:
            api_key = os.getenv("GROK_API_KEY")
            base_url = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
            if not api_key:
                print("WARNING: GROK_API_KEY not found. Summarization will be skipped.")
                return None
            
            self._llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model="grok-2-latest", # Fast and capable
                temperature=0.0
            )
        return self._llm

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

    def generate_summary(self, content: str, filename: str) -> str:
        """Uses LLM to generate a one-sentence summary."""
        if not self.llm:
            return "No summary (LLM not configured)."
        
        try:
            # Truncate content to avoid huge context costs (first 3000 chars is usually enough for a summary)
            truncated_content = content[:3000]
            
            prompt = ChatPromptTemplate.from_template(
                "Summarize the following document in ONE concise sentence describing what it is. "
                "Focus on its purpose (e.g., 'A contract for...').\n\nFilename: {filename}\nContent:\n{content}"
            )
            chain = prompt | self.llm
            response = chain.invoke({"filename": filename, "content": truncated_content})
            return response.content.strip()
        except Exception as e:
            print(f"  Summary generation failed: {e}")
            return "Summary generation failed."

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
                    
                    # 1. Generate Summary (The New Step)
                    print(f"  [{i+1}/{len(files_to_process)}] Ingesting: {os.path.basename(file_path)}")
                    summary = self.generate_summary(content, os.path.basename(file_path))
                    
                    # 2. Update State with Summary
                    current_state[file_path]["summary"] = summary
                    
                    # 3. Embed and Save to Vector DB
                    chunks = self.text_splitter.split_documents(docs)
                    if chunks:
                        # Optional: Enhance metadata with summary?
                        # for c in chunks: c.metadata["summary"] = summary
                        db.add_documents(chunks)
                        
                except Exception as e:
                    print(f"  FAILED to process {file_path}: {e}")

        # -- C. COMMIT STATE --
        print("\nSaving new state to ledger...")
        self.save_state(current_state)
        print("SYNC COMPLETE! Database is now identical to source.")


if __name__ == "__main__":
    SOURCE_PATH = os.getenv("RAG_SOURCE_PATH", "/mnt/e/WMS_selection")
    DB_PATH = os.getenv("RAG_DB_PATH", "/mnt/e/chroma_db_wms")
    
    print("--- WMS RAG INGESTION SYSTEM (With AI Summaries) ---")
    print(f"Source: {SOURCE_PATH}")
    print(f"Database: {DB_PATH}")
    
    if not os.path.exists(SOURCE_PATH):
        print(f"ERROR: Source path '{SOURCE_PATH}' does not exist.")
        exit(1)
        
    system = IngestionSystem(source_path=SOURCE_PATH, db_path=DB_PATH)
    system.run()
