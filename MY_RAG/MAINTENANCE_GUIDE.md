# Knowledge Base Maintenance Guide

This guide explains how to keep your Vector Database in sync with your Google Drive data.

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

### Step 2: Sync SSD (E:) to Vector Database
**Goal:** Update the AI brains to match your files on E:.

1. Open WSL (Ubuntu).
2. Navigate to your project:
   ```bash
   cd ~/projects/llm_engineering
   ```
3. Open the Incremental Sync Notebook:
   *   Filename: `week5/day2-incremental-sync.ipynb`
4. **Run All Cells.**

### Summary of Logic
| Changed on Google Drive? | Step 1 (Robocopy) | Step 2 (Python Script) | Result |
| :--- | :--- | :--- | :--- |
| **New File** | Downloads to E: | Detects new file $\to$ Add to DB | **Added** |
| **Edited File** | Re-downloads to E: | Detects hash change $\to$ Update DB | **Updated** |
| **Deleted File** | Deletes from E: | Detects missing file $\to$ Remove from DB | **Deleted** |
| **No Change** | **Skips (Fast)** | **Skips (Fast)** | **No Action** |
