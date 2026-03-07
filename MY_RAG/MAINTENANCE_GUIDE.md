# Knowledge Base Maintenance Guide

This guide explains how to keep your Vector Database in sync with your Google Drive data using the new V2 Agentic Pipeline.

## The 2-Step Sync Pipeline

To update your AI, you must perform these two steps in order.

### Step 1: Sync Google Drive (G:) to Local SSD (E:)
**Goal:** Make your E: drive an exact mirror of the Cloud, downloading ONLY what changed.

1. Open **Command Prompt** or **PowerShell** in Windows.
2. Run the following command (Copy/Paste this):

```powershell
robocopy "G:\Shared drives\WMS selection" "E:\WMS_selection" /MIR /MT:8 /R:1 /W:1 /FFT /XF *.gdoc *.gsheet *.gslides *.gdraw *.gtable *.gform *.gmap
```

**What do these flags do?**
*   `/MIR` **(Mirror):** Copies new files, updates changed files, and **DELETES** files on E: that were deleted from G:. (True synchronization).
*   `/MT:8`: **(Multi-Thread):** Copies 8 files at once (faster).
*   `/R:1 /W:1`: If a file fails (e.g., Google Drive glitch), retry once, wait 1 second, then skip. Prevents getting stuck.
*   `/FFT`: **(Fat File Time):** Allows for 2-second timestamp differences (important when syncing between different drive types).
*   `/XF *.gdoc ...`: **(Exclude Files):** Skips native Google Docs/Sheets files. These are just web links (not real files) and cause "Invalid MS-DOS function" errors if you try to copy them.

---

### Step 2: Agentic Schema Ingestion (Native RAG V2)
**Goal:** Process targeted folders, convert multi-format docs to Markdown, pre-screen them with an LLM, and embed into ChromaDB.

1. Open WSL (Ubuntu).
2. Navigate to your project's RAG directory:
   ```bash
   cd ~/projects/llm_engineering/MY_RAG
   ```
3. **Configure the Ingestion Run:** Open `config.json` and verify the targeted runtime settings:
   *   `included_folders`: Define exactly which folders to ingest (e.g., `["/mnt/e/WMS_selection/WMS Business Process and SOP/SYML SOP"]`).
   *   `llm_model`: Set the "Brain" using LiteLLM prefixes (e.g., `xai/grok-4-1-fast-non-reasoning`). Ensure `.env` has the applicable API KEY.
   *   `max_workers`: Controls the number of threads for parallel document parsing and AI screening calls.

4. **Execute the Agentic Pipeline:**
   This project uses `uv` for seamless dependency management via PEP 723 inline metadata. Just run:
   ```bash
   uv run ingest_v2.py
   ```

### V2 Processing Pipeline Flow
When you run `ingest_v2.py`, the python script natively handles the following workflow:
1. **Format Conversion (`pymupdf4llm` & `pandas`):** Perfect transcriptions of PDFs to Markdown (preserving tables/structure) alongside standard `.md` and `.txt` files. Automatically converts Excel Workbooks (`.xlsx`) sheet-by-sheet into markdown tables.
2. **Metadata Screening (`litellm`):** The LLM agent acts as a "Senior Technical Screener", evaluating the document and generating a semantic summary.
3. **Native Pydantic Chunking:** Splits the text and stamps the AI's semantic summary into the structured metadata of *every single chunk* so context is never lost during retrieval.
4. **Local GPU Embedding (`sentence-transformers`):** Embeds the text using your `gte-Qwen2-7B-instruct` local model.
5. **Persistence (`chromadb`):** Writes the chunks natively to Chroma without the bloated LangChain wrappers.
