try:
    from langchain_core.language_models import ModelProfile
    print("ModelProfile found!")
except ImportError as e:
    print(f"ImportError: {e}")
    import langchain_core.language_models
    print(f"Available in langchain_core.language_models: {dir(langchain_core.language_models)}")
