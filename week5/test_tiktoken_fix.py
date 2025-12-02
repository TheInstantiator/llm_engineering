import tiktoken

MODEL = "grok-4-1-fast-reasoning"
print(f"Testing tokenizer for model: {MODEL}")

try:
    encoding = tiktoken.encoding_for_model(MODEL)
    print("Success: Found specific encoding.")
except KeyError:
    print("Caught expected KeyError (model not found).")
    print("Falling back to 'cl100k_base'...")
    encoding = tiktoken.get_encoding("cl100k_base")

print(f"Encoding loaded: {encoding.name}")
tokens = encoding.encode("Hello world")
print(f"Test encoding 'Hello world': {tokens}")
