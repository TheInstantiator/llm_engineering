import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Load Environment Variables
load_dotenv(override=True)
key = os.getenv("GEMINI_API_KEY")

print(f"1. Checking GEMINI_API_KEY...")
if not key:
    print("   ERROR: GEMINI_API_KEY not found in environment variables.")
else:
    print(f"   SUCCESS: Found key starting with: {key[:4]}...")

# 2. Test Direct API Call (Bypassing LangChain)
print("\n2. Testing Direct Google API Call...")
if key:
    genai.configure(api_key=key)
    try:
        models = list(genai.list_models())
        print(f"   SUCCESS: API Key accepted. Found {len(models)} models.")
    except Exception as e:
        print(f"   ERROR: Direct API call failed: {e}")

# 3. Test LangChain Embeddings
print("\n3. Testing LangChain Embeddings...")
if key:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=key
        )
        vector = embeddings.embed_query("hello world")
        print(f"   SUCCESS: Embeddings generated. Vector length: {len(vector)}")
    except Exception as e:
        print(f"   ERROR: LangChain embedding failed: {e}")
