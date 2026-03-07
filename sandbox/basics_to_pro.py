class BaseProcessor:
    """The 'Abstract' base class (Java: abstract class)."""
    def __init__(self, items):
        self.items = items
    
    def process(self):
        """A base method to be overridden."""
        raise NotImplementedError("Subclasses must implement process()")

class DataManager(BaseProcessor):
    """Subclass inheriting from BaseProcessor (Java: extends)."""
    
    def __init__(self, items, label="Default"):
        # Call the parent constructor (Java: super(items))
        super().__init__(items)
        self.label = label
    
    @staticmethod
    def _clean_text(text):
        """An internal utility (Java: private/protected static)."""
        return text.strip().upper()

    def process(self):
        """Implementing the specific logic (Java: @Override)."""
        print(f"--- Running {self.label} Manager ---")
        return [self._clean_text(i) for i in self.items if len(i) > 3]

def main():
    print("🚀 Starting Refactored OOP Script!")
    
    fruits = [" apple ", "cat", " banana"]
    
    # We use the subclass
    manager = DataManager(fruits, label="Fruit Processor")
    results = manager.process()
    
    print(f"Final Results: {results}")

# This is the "Magic Block"
if __name__ == "__main__":
    # This only runs if we call this file directly
    main()
