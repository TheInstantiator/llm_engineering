Week 5: Vectors and Embeddings – The Foundation of RAG
Study Guide Notes
Compiled from lecture transcript – Day 1 of Vectors Week
📅 Date: January 2026
🧠 Goal: Build intuition for vector embeddings, understand how they enable “fuzzy” semantic lookup, and see why they are the core idea behind Retrieval-Augmented Generation (RAG).
1. Two Types of Language Models
There are two fundamental flavors of LLMs:


























TypeNameCore TaskExamples We’ve Used So FarKey Characteristic🔄 AutoregressiveDecoder-only / Causal LMPredict the next token given previous tokens (one token at a time)GPT-4o, Claude, Gemini, Grok, etc.Generates text sequentially🔍 AutoencodingEncoder / Embedding ModelTake a full input sequence and produce a single output that captures the entire meaningBERT, OpenAI embeddings, all-MiniLMProduces a fixed-size representation (vector)
Key Insight 🚀
Autoregressive models are what we usually call “LLMs” – they generate responses.
Encoder (embedding) models are the secret sauce behind semantic search and RAG.
2. Tokens vs. Vectors – Clear Demystification























ConceptRoleDescriptionExample🗝️ TokensInputSimple numeric IDs that represent pieces of text (subwords via BPE, etc.)"Heathrow" → [12345, 678]📊 VectorsOutputDense floating-point arrays that capture semantic meaning after processing by the model"Heathrow" → [-0.012, 0.543, …, 0.221] (1536-dim for text-embedding-3-large)
Flow
Text → Tokenizer → Tokens → Embedding Model → Vector Embedding
Even autoregressive models internally convert tokens to vectors layer-by-layer, but we only expose the final vector from encoder models.
3. How Vectors Represent Meaning
Vectors live in high-dimensional space (typically 384–3072 dimensions).
Core Properties

Proximity = Semantic Similarity 🌐
Points close together (measured by cosine similarity) have similar meaning, even with different words.#mermaid-diagram-mermaid-b44e1jg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#ccc;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#mermaid-diagram-mermaid-b44e1jg .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#mermaid-diagram-mermaid-b44e1jg .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#mermaid-diagram-mermaid-b44e1jg .error-icon{fill:#a44141;}#mermaid-diagram-mermaid-b44e1jg .error-text{fill:#ddd;stroke:#ddd;}#mermaid-diagram-mermaid-b44e1jg .edge-thickness-normal{stroke-width:1px;}#mermaid-diagram-mermaid-b44e1jg .edge-thickness-thick{stroke-width:3.5px;}#mermaid-diagram-mermaid-b44e1jg .edge-pattern-solid{stroke-dasharray:0;}#mermaid-diagram-mermaid-b44e1jg .edge-thickness-invisible{stroke-width:0;fill:none;}#mermaid-diagram-mermaid-b44e1jg .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-diagram-mermaid-b44e1jg .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-diagram-mermaid-b44e1jg .marker{fill:lightgrey;stroke:lightgrey;}#mermaid-diagram-mermaid-b44e1jg .marker.cross{stroke:lightgrey;}#mermaid-diagram-mermaid-b44e1jg svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-diagram-mermaid-b44e1jg p{margin:0;}#mermaid-diagram-mermaid-b44e1jg .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#ccc;}#mermaid-diagram-mermaid-b44e1jg .cluster-label text{fill:#F9FFFE;}#mermaid-diagram-mermaid-b44e1jg .cluster-label span{color:#F9FFFE;}#mermaid-diagram-mermaid-b44e1jg .cluster-label span p{background-color:transparent;}#mermaid-diagram-mermaid-b44e1jg .label text,#mermaid-diagram-mermaid-b44e1jg span{fill:#ccc;color:#ccc;}#mermaid-diagram-mermaid-b44e1jg .node rect,#mermaid-diagram-mermaid-b44e1jg .node circle,#mermaid-diagram-mermaid-b44e1jg .node ellipse,#mermaid-diagram-mermaid-b44e1jg .node polygon,#mermaid-diagram-mermaid-b44e1jg .node path{fill:#1f2020;stroke:#ccc;stroke-width:1px;}#mermaid-diagram-mermaid-b44e1jg .rough-node .label text,#mermaid-diagram-mermaid-b44e1jg .node .label text,#mermaid-diagram-mermaid-b44e1jg .image-shape .label,#mermaid-diagram-mermaid-b44e1jg .icon-shape .label{text-anchor:middle;}#mermaid-diagram-mermaid-b44e1jg .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#mermaid-diagram-mermaid-b44e1jg .rough-node .label,#mermaid-diagram-mermaid-b44e1jg .node .label,#mermaid-diagram-mermaid-b44e1jg .image-shape .label,#mermaid-diagram-mermaid-b44e1jg .icon-shape .label{text-align:center;}#mermaid-diagram-mermaid-b44e1jg .node.clickable{cursor:pointer;}#mermaid-diagram-mermaid-b44e1jg .root .anchor path{fill:lightgrey!important;stroke-width:0;stroke:lightgrey;}#mermaid-diagram-mermaid-b44e1jg .arrowheadPath{fill:lightgrey;}#mermaid-diagram-mermaid-b44e1jg .edgePath .path{stroke:lightgrey;stroke-width:2.0px;}#mermaid-diagram-mermaid-b44e1jg .flowchart-link{stroke:lightgrey;fill:none;}#mermaid-diagram-mermaid-b44e1jg .edgeLabel{background-color:hsl(0, 0%, 34.4117647059%);text-align:center;}#mermaid-diagram-mermaid-b44e1jg .edgeLabel p{background-color:hsl(0, 0%, 34.4117647059%);}#mermaid-diagram-mermaid-b44e1jg .edgeLabel rect{opacity:0.5;background-color:hsl(0, 0%, 34.4117647059%);fill:hsl(0, 0%, 34.4117647059%);}#mermaid-diagram-mermaid-b44e1jg .labelBkg{background-color:rgba(87.75, 87.75, 87.75, 0.5);}#mermaid-diagram-mermaid-b44e1jg .cluster rect{fill:hsl(180, 1.5873015873%, 28.3529411765%);stroke:rgba(255, 255, 255, 0.25);stroke-width:1px;}#mermaid-diagram-mermaid-b44e1jg .cluster text{fill:#F9FFFE;}#mermaid-diagram-mermaid-b44e1jg .cluster span{color:#F9FFFE;}#mermaid-diagram-mermaid-b44e1jg div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(20, 1.5873015873%, 12.3529411765%);border:1px solid rgba(255, 255, 255, 0.25);border-radius:2px;pointer-events:none;z-index:100;}#mermaid-diagram-mermaid-b44e1jg .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#ccc;}#mermaid-diagram-mermaid-b44e1jg rect.text{fill:none;stroke-width:0;}#mermaid-diagram-mermaid-b44e1jg .icon-shape,#mermaid-diagram-mermaid-b44e1jg .image-shape{background-color:hsl(0, 0%, 34.4117647059%);text-align:center;}#mermaid-diagram-mermaid-b44e1jg .icon-shape p,#mermaid-diagram-mermaid-b44e1jg .image-shape p{background-color:hsl(0, 0%, 34.4117647059%);padding:2px;}#mermaid-diagram-mermaid-b44e1jg .icon-shape rect,#mermaid-diagram-mermaid-b44e1jg .image-shape rect{opacity:0.5;background-color:hsl(0, 0%, 34.4117647059%);fill:hsl(0, 0%, 34.4117647059%);}#mermaid-diagram-mermaid-b44e1jg :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}High-Dimensional Spacevery close'Ticket prices to London'Vector A'Flight cost from JFK to Heathrow'Vector B
Vector Arithmetic Captures Relationships 🧮
Famous example (discovered in Word2Vec, still works today):textking - man + woman ≈ queen
Paris - France + Italy ≈ Rome#mermaid-diagram-mermaid-ica1upb{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#ccc;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#mermaid-diagram-mermaid-ica1upb .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#mermaid-diagram-mermaid-ica1upb .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#mermaid-diagram-mermaid-ica1upb .error-icon{fill:#a44141;}#mermaid-diagram-mermaid-ica1upb .error-text{fill:#ddd;stroke:#ddd;}#mermaid-diagram-mermaid-ica1upb .edge-thickness-normal{stroke-width:1px;}#mermaid-diagram-mermaid-ica1upb .edge-thickness-thick{stroke-width:3.5px;}#mermaid-diagram-mermaid-ica1upb .edge-pattern-solid{stroke-dasharray:0;}#mermaid-diagram-mermaid-ica1upb .edge-thickness-invisible{stroke-width:0;fill:none;}#mermaid-diagram-mermaid-ica1upb .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-diagram-mermaid-ica1upb .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-diagram-mermaid-ica1upb .marker{fill:lightgrey;stroke:lightgrey;}#mermaid-diagram-mermaid-ica1upb .marker.cross{stroke:lightgrey;}#mermaid-diagram-mermaid-ica1upb svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-diagram-mermaid-ica1upb p{margin:0;}#mermaid-diagram-mermaid-ica1upb .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#ccc;}#mermaid-diagram-mermaid-ica1upb .cluster-label text{fill:#F9FFFE;}#mermaid-diagram-mermaid-ica1upb .cluster-label span{color:#F9FFFE;}#mermaid-diagram-mermaid-ica1upb .cluster-label span p{background-color:transparent;}#mermaid-diagram-mermaid-ica1upb .label text,#mermaid-diagram-mermaid-ica1upb span{fill:#ccc;color:#ccc;}#mermaid-diagram-mermaid-ica1upb .node rect,#mermaid-diagram-mermaid-ica1upb .node circle,#mermaid-diagram-mermaid-ica1upb .node ellipse,#mermaid-diagram-mermaid-ica1upb .node polygon,#mermaid-diagram-mermaid-ica1upb .node path{fill:#1f2020;stroke:#ccc;stroke-width:1px;}#mermaid-diagram-mermaid-ica1upb .rough-node .label text,#mermaid-diagram-mermaid-ica1upb .node .label text,#mermaid-diagram-mermaid-ica1upb .image-shape .label,#mermaid-diagram-mermaid-ica1upb .icon-shape .label{text-anchor:middle;}#mermaid-diagram-mermaid-ica1upb .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#mermaid-diagram-mermaid-ica1upb .rough-node .label,#mermaid-diagram-mermaid-ica1upb .node .label,#mermaid-diagram-mermaid-ica1upb .image-shape .label,#mermaid-diagram-mermaid-ica1upb .icon-shape .label{text-align:center;}#mermaid-diagram-mermaid-ica1upb .node.clickable{cursor:pointer;}#mermaid-diagram-mermaid-ica1upb .root .anchor path{fill:lightgrey!important;stroke-width:0;stroke:lightgrey;}#mermaid-diagram-mermaid-ica1upb .arrowheadPath{fill:lightgrey;}#mermaid-diagram-mermaid-ica1upb .edgePath .path{stroke:lightgrey;stroke-width:2.0px;}#mermaid-diagram-mermaid-ica1upb .flowchart-link{stroke:lightgrey;fill:none;}#mermaid-diagram-mermaid-ica1upb .edgeLabel{background-color:hsl(0, 0%, 34.4117647059%);text-align:center;}#mermaid-diagram-mermaid-ica1upb .edgeLabel p{background-color:hsl(0, 0%, 34.4117647059%);}#mermaid-diagram-mermaid-ica1upb .edgeLabel rect{opacity:0.5;background-color:hsl(0, 0%, 34.4117647059%);fill:hsl(0, 0%, 34.4117647059%);}#mermaid-diagram-mermaid-ica1upb .labelBkg{background-color:rgba(87.75, 87.75, 87.75, 0.5);}#mermaid-diagram-mermaid-ica1upb .cluster rect{fill:hsl(180, 1.5873015873%, 28.3529411765%);stroke:rgba(255, 255, 255, 0.25);stroke-width:1px;}#mermaid-diagram-mermaid-ica1upb .cluster text{fill:#F9FFFE;}#mermaid-diagram-mermaid-ica1upb .cluster span{color:#F9FFFE;}#mermaid-diagram-mermaid-ica1upb div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(20, 1.5873015873%, 12.3529411765%);border:1px solid rgba(255, 255, 255, 0.25);border-radius:2px;pointer-events:none;z-index:100;}#mermaid-diagram-mermaid-ica1upb .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#ccc;}#mermaid-diagram-mermaid-ica1upb rect.text{fill:none;stroke-width:0;}#mermaid-diagram-mermaid-ica1upb .icon-shape,#mermaid-diagram-mermaid-ica1upb .image-shape{background-color:hsl(0, 0%, 34.4117647059%);text-align:center;}#mermaid-diagram-mermaid-ica1upb .icon-shape p,#mermaid-diagram-mermaid-ica1upb .image-shape p{background-color:hsl(0, 0%, 34.4117647059%);padding:2px;}#mermaid-diagram-mermaid-ica1upb .icon-shape rect,#mermaid-diagram-mermaid-ica1upb .image-shape rect{opacity:0.5;background-color:hsl(0, 0%, 34.4117647059%);fill:hsl(0, 0%, 34.4117647059%);}#mermaid-diagram-mermaid-ica1upb :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}subtract VMadd VWKingVectorManVectorWomanVectorQueenVectorTempThis works at sentence/paragraph level with modern embedding models too.

4. The Big Idea: RAG Powered by Vectors
Problem with naive search
User asks: “What are ticket prices to Heathrow?”
Database contains: “Ticket prices to London”
→ Keyword search fails.
Vector solution

Pre-compute embeddings for all database chunks → store in vector database
At query time: embed the user question
Retrieve chunks whose vectors are closest (highest cosine similarity)
Insert retrieved text (natural language) into prompt for autoregressive LLM

#mermaid-diagram-mermaid-thsvady{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#ccc;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#mermaid-diagram-mermaid-thsvady .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#mermaid-diagram-mermaid-thsvady .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#mermaid-diagram-mermaid-thsvady .error-icon{fill:#a44141;}#mermaid-diagram-mermaid-thsvady .error-text{fill:#ddd;stroke:#ddd;}#mermaid-diagram-mermaid-thsvady .edge-thickness-normal{stroke-width:1px;}#mermaid-diagram-mermaid-thsvady .edge-thickness-thick{stroke-width:3.5px;}#mermaid-diagram-mermaid-thsvady .edge-pattern-solid{stroke-dasharray:0;}#mermaid-diagram-mermaid-thsvady .edge-thickness-invisible{stroke-width:0;fill:none;}#mermaid-diagram-mermaid-thsvady .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-diagram-mermaid-thsvady .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-diagram-mermaid-thsvady .marker{fill:lightgrey;stroke:lightgrey;}#mermaid-diagram-mermaid-thsvady .marker.cross{stroke:lightgrey;}#mermaid-diagram-mermaid-thsvady svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-diagram-mermaid-thsvady p{margin:0;}#mermaid-diagram-mermaid-thsvady .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#ccc;}#mermaid-diagram-mermaid-thsvady .cluster-label text{fill:#F9FFFE;}#mermaid-diagram-mermaid-thsvady .cluster-label span{color:#F9FFFE;}#mermaid-diagram-mermaid-thsvady .cluster-label span p{background-color:transparent;}#mermaid-diagram-mermaid-thsvady .label text,#mermaid-diagram-mermaid-thsvady span{fill:#ccc;color:#ccc;}#mermaid-diagram-mermaid-thsvady .node rect,#mermaid-diagram-mermaid-thsvady .node circle,#mermaid-diagram-mermaid-thsvady .node ellipse,#mermaid-diagram-mermaid-thsvady .node polygon,#mermaid-diagram-mermaid-thsvady .node path{fill:#1f2020;stroke:#ccc;stroke-width:1px;}#mermaid-diagram-mermaid-thsvady .rough-node .label text,#mermaid-diagram-mermaid-thsvady .node .label text,#mermaid-diagram-mermaid-thsvady .image-shape .label,#mermaid-diagram-mermaid-thsvady .icon-shape .label{text-anchor:middle;}#mermaid-diagram-mermaid-thsvady .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#mermaid-diagram-mermaid-thsvady .rough-node .label,#mermaid-diagram-mermaid-thsvady .node .label,#mermaid-diagram-mermaid-thsvady .image-shape .label,#mermaid-diagram-mermaid-thsvady .icon-shape .label{text-align:center;}#mermaid-diagram-mermaid-thsvady .node.clickable{cursor:pointer;}#mermaid-diagram-mermaid-thsvady .root .anchor path{fill:lightgrey!important;stroke-width:0;stroke:lightgrey;}#mermaid-diagram-mermaid-thsvady .arrowheadPath{fill:lightgrey;}#mermaid-diagram-mermaid-thsvady .edgePath .path{stroke:lightgrey;stroke-width:2.0px;}#mermaid-diagram-mermaid-thsvady .flowchart-link{stroke:lightgrey;fill:none;}#mermaid-diagram-mermaid-thsvady .edgeLabel{background-color:hsl(0, 0%, 34.4117647059%);text-align:center;}#mermaid-diagram-mermaid-thsvady .edgeLabel p{background-color:hsl(0, 0%, 34.4117647059%);}#mermaid-diagram-mermaid-thsvady .edgeLabel rect{opacity:0.5;background-color:hsl(0, 0%, 34.4117647059%);fill:hsl(0, 0%, 34.4117647059%);}#mermaid-diagram-mermaid-thsvady .labelBkg{background-color:rgba(87.75, 87.75, 87.75, 0.5);}#mermaid-diagram-mermaid-thsvady .cluster rect{fill:hsl(180, 1.5873015873%, 28.3529411765%);stroke:rgba(255, 255, 255, 0.25);stroke-width:1px;}#mermaid-diagram-mermaid-thsvady .cluster text{fill:#F9FFFE;}#mermaid-diagram-mermaid-thsvady .cluster span{color:#F9FFFE;}#mermaid-diagram-mermaid-thsvady div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(20, 1.5873015873%, 12.3529411765%);border:1px solid rgba(255, 255, 255, 0.25);border-radius:2px;pointer-events:none;z-index:100;}#mermaid-diagram-mermaid-thsvady .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#ccc;}#mermaid-diagram-mermaid-thsvady rect.text{fill:none;stroke-width:0;}#mermaid-diagram-mermaid-thsvady .icon-shape,#mermaid-diagram-mermaid-thsvady .image-shape{background-color:hsl(0, 0%, 34.4117647059%);text-align:center;}#mermaid-diagram-mermaid-thsvady .icon-shape p,#mermaid-diagram-mermaid-thsvady .image-shape p{background-color:hsl(0, 0%, 34.4117647059%);padding:2px;}#mermaid-diagram-mermaid-thsvady .icon-shape rect,#mermaid-diagram-mermaid-thsvady .image-shape rect{opacity:0.5;background-color:hsl(0, 0%, 34.4117647059%);fill:hsl(0, 0%, 34.4117647059%);}#mermaid-diagram-mermaid-thsvady :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}Nearest NeighborsUser Question
“Ticket prices to Heathrow”Embedding ModelKnowledge Base ChunksEmbedding Model
(pre-computed)(Vector Database
e.g., Chroma, Pinecone)Query VectorRetrieved Text
“Ticket prices to London...”Prompt to Autoregressive LLMAccurate Answer
Critical Clarification ⚠️
The embedding model is completely separate from the generative LLM.
Only natural language text goes into the final prompt – never raw vectors.
5. Popular Embedding Models (as of 2026)









































ModelProviderDimensionsNotes / When to Usetext-embedding-3-largeOpenAI3072Highest quality, great for RAGtext-embedding-3-smallOpenAI1536Faster, cheaper, still excellenttext-embedding-ada-002 (legacy)OpenAI1536Still widely used but superseded by v3all-MiniLM-L6-v2Hugging Face384Fast, open-source, runs locallyBGE-large-en-v1.5Hugging Face1024Current SOTA open-source (2025–2026 leaderboard)
Modern Best Practice Callout 📌
As of early 2026, text-embedding-3-large remains OpenAI’s flagship. For open-source, models like BGE or Snowflake/snowflake-arctic-embed often outperform all-MiniLM on MTEB benchmarks. Always check the latest Massive Text Embedding Benchmark (MTEB) leaderboard on Hugging Face.
6. Code Examples
6.1 OpenAI Embeddings (Proprietary)
Python# pip install openai
from openai import OpenAI
client = OpenAI()  # uses your OPENAI_API_KEY env var

def get_embedding(text: str, model: str = "text-embedding-3-large"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Example usage
query_emb = get_embedding("What are ticket prices to Heathrow?")
doc_emb = get_embedding("Flight costs from New York to London Heathrow are...")
print(len(query_emb))  # → 3072 (for large)
6.2 Open-Source Embeddings (Sentence-Transformers)
Python# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # or "BAAI/bge-large-en-v1.5"

def get_embedding(text: str):
    return model.encode(text, normalize_embeddings=True)  # returns numpy array

query_emb = get_embedding("What are ticket prices to Heathrow?")
doc_emb   = get_embedding("Ticket prices to London")
Cosine Similarity Calculation
Pythonfrom sklearn.metrics.pairwise import cosine_similarity
import numpy as np

similarity = cosine_similarity([query_emb], [doc_emb])[0][0]
print(f"Similarity: {similarity:.4f}")  # closer to 1.0 = more similar
7. Study Questions
Q1. What is the fundamental difference between autoregressive and autoencoding (encoder) LLMs?
Answer:
Autoregressive models predict the next token sequentially. Encoder models take a complete input and produce a fixed representation (vector embedding) that captures the overall meaning.
Q2. Tokens are inputs; vectors are outputs. True or false? Explain.
Answer:
True. Tokens are numeric IDs produced by the tokenizer for model input. Vectors are dense floating-point representations produced by an embedding/encoder model that capture semantic meaning.
Q3. Why does vector proximity enable “fuzzy” lookup in RAG?
Answer:
Texts with similar meaning (even different wording) map to nearby points in vector space. Retrieving nearest neighbors therefore returns semantically relevant chunks, not just keyword matches.
Q4. In the classic vector arithmetic example, what does king − man + woman approximately equal?
Answer:
queen. This demonstrates that embedding spaces capture relational analogies.
Q5. In a RAG pipeline, do raw vectors ever get sent to the final generative LLM? Why or why not?
Answer:
No. Only the retrieved natural-language text is inserted into the prompt. Generative LLMs expect tokenized text, not raw numeric vectors.
Q6. Name three popular embedding models and one situation where you’d choose an open-source model over OpenAI’s.
Answer:
Models: text-embedding-3-large (OpenAI), all-MiniLM-L6-v2, BGE-large-en-v1.5 (both HF).
Choose open-source when you need offline/local inference, lower cost at scale, or full control over data privacy.
Q7. What metric is most commonly used to measure distance between embedding vectors?
Answer:
Cosine similarity (equivalent to finding nearest neighbors in normalized embedding space).