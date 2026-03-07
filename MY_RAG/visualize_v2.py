import argparse
import os
import json
import numpy as np
import chromadb
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.io as pio

class VectorVisualizer:
    """
    A class to handle retrieving data from ChromaDB and 
    generating a t-SNE visualization.
    
    In Java terms, think of this like a 'Service' or 'Repository' class 
    that handles the business logic for connecting to the database and 
    processing the data for the frontend.
    """
    
    def __init__(self, config_path: str = "config.json"):
        # 1. Load configuration 
        # (__init__ is Python's version of a Java Constructor)
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.db_path = self.config.get("db_path", "./chroma_db_v2")
        self.collection_name = "wms_docs_v2"
        
        print(f"Connecting to ChromaDB at: {self.db_path}")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_collection(name=self.collection_name)
    
    def fetch_data(self):
        """Retrieves embeddings, text, and metadata from the store."""
        print(f"Fetching vectors from '{self.collection_name}'...")
        # include=['embeddings', 'documents', 'metadatas'] is the key here
        results = self.collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        if len(results['embeddings']) == 0:
            raise ValueError("No embeddings found in the collection. Did you run the ingestion yet?")
            
        return results

    def generate_visualization(self, output_html="vector_viz.html", dimensions=3):
        """Performs t-SNE and saves an interactive Plotly HTML file in 2D or 3D."""
        data = self.fetch_data()
        
        vectors = np.array(data['embeddings'])
        documents = data['documents']
        metadatas = data['metadatas']
        
        # 2. t-SNE Magic (Reducing High-D vectors to 2D or 3D)
        # t-SNE takes data with hundreds of dimensions (like our embedding vectors) 
        # and figures out how to plot them in 2D/3D space while keeping similar items close together!
        print(f"Running t-SNE dimensionality reduction to {dimensions}D (this might take a moment)...")
        tsne = TSNE(n_components=dimensions, random_state=42, init='pca', learning_rate='auto')
        reduced_vectors = tsne.fit_transform(vectors)
        
        # 3. Prepare labels for the plot
        hover_text = []
        for meta, doc in zip(metadatas, documents):
            path = meta.get('file_path', 'Unknown')
            snippet = doc[:100].replace('\n', ' ')
            hover_text.append(f"File: {os.path.basename(path)}<br>Snippet: {snippet}...")

        # 4. Create the Plotly Figure
        print("Creating interactive plot...")
        if dimensions == 3:
            trace = go.Scatter3d(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                z=reduced_vectors[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    opacity=0.7,
                    line=dict(width=0.5, color='DarkSlateGrey'),
                    color=np.random.randn(len(reduced_vectors)),
                    colorscale='Viridis',
                ),
                text=hover_text,
                hoverinfo='text'
            )
            layout_args = dict(
                scene=dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    zaxis_title='Dimension 3'
                )
            )
        else:
            trace = go.Scatter(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                mode='markers',
                marker=dict(
                    size=7,
                    opacity=0.7,
                    line=dict(width=1, color='DarkSlateGrey'),
                    color=np.random.randn(len(reduced_vectors)),
                    colorscale='Viridis',
                ),
                text=hover_text,
                hoverinfo='text'
            )
            layout_args = dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
            )

        fig = go.Figure(data=[trace])

        fig.update_layout(
            title=f"{dimensions}D Projection of {self.collection_name} (t-SNE)",
            template="plotly_dark",
            width=1000,
            height=700,
            **layout_args
        )

        # 5. Export to HTML
        # In a .py script, we can't easily pop up a window in all environments,
        # so saving to HTML is the most reliable way to view interactive plots.
        pio.write_html(fig, file=output_html, auto_open=False)
        print(f"✅ Success! Visualization saved to: {os.path.abspath(output_html)}")



# ... (keep existing imports, handled below in the diff automatically since we target the end of file)
def main():
    """Main execution block with argument parsing."""
    
    # Argparse handles command-line arguments. In Java, this is what you'd do 
    # manually by looking at the 'String[] args' parameter in public static void main.
    # Python makes it easier by auto-generating a --help menu.
    parser = argparse.ArgumentParser(description="Generate 2D or 3D vector visualizations.")
    parser.add_argument(
        "--dim", 
        type=int, 
        choices=[0, 2, 3], 
        default=0, 
        help="Dimension output: 2 for 2D, 3 for 3D, 0 for both (default: 0)"
    )
    args = parser.parse_args()

    try:
        # Safely resolve paths relative to where the script is saved!
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        viz = VectorVisualizer(config_path)

        if args.dim in (0, 2):
            output_2d = os.path.join(script_dir, "rag_visualization_2d.html")
            viz.generate_visualization(output_2d, dimensions=2)
            
        if args.dim in (0, 3):
            output_3d = os.path.join(script_dir, "rag_visualization_3d.html")
            viz.generate_visualization(output_3d, dimensions=3)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
