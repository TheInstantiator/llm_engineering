from dotenv import load_dotenv
import os

# Load environment variables
loaded = load_dotenv()
print(f"Dotenv loaded: {loaded}")

# Check for GROK_BASE_URL
grok_url = os.getenv("GROK_BASE_URL")
print(f"GROK_BASE_URL: {grok_url}")

if grok_url:
    print("SUCCESS: Environment variable found.")
else:
    print("FAILURE: Environment variable NOT found.")
