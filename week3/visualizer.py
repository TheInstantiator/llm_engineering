import os
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict
import math
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

class TokenPredictor:
    def __init__(self, model_provider: str, model_name: str):
        self.model_name = model_name
        api_key = None
        base_url = None

        if model_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif model_provider == "grok":
            api_key = os.getenv("GROK_API_KEY")
            base_url = os.getenv("GROK_URL")
        elif model_provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            base_url = os.getenv("GEMINI_URL")
        elif model_provider == "ollama":
            api_key = "ollama"
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        
        print(api_key[:4])
        print(base_url)
        print(self.model_name)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def predict_tokens(self, prompt: str, max_tokens: int = 100) -> List[Dict]:
        """
        Generate text token by token and track prediction probabilities.
        Returns list of predictions with top token and alternatives.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,  # Use temperature 0 for deterministic output
            logprobs=True,
            seed=42,
            top_logprobs=3,  # Get top 3 token predictions
            stream=True,  # Stream the response
        )

        predictions = []
        for chunk in response:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                logprobs = chunk.choices[0].logprobs.content[0].top_logprobs
                logprob_dict = {item.token: item.logprob for item in logprobs}

                # Get top predicted token and probability
                top_token = token
                top_prob = logprob_dict.get(token, -100) # Use get for safety

                # Get alternative predictions
                alternatives = []
                for alt_token, alt_prob in logprob_dict.items():
                    if alt_token != token:
                        alternatives.append((alt_token, math.exp(alt_prob)))
                alternatives.sort(key=lambda x: x[1], reverse=True)

                prediction = {
                    "token": top_token,
                    "probability": math.exp(top_prob),
                    "alternatives": alternatives[:2],  # Keep top 2 alternatives
                }
                predictions.append(prediction)

        return predictions


def create_token_graph(model_name: str, predictions: List[Dict]) -> nx.DiGraph:
    """
    Create a directed graph showing token predictions and alternatives.
    """
    G = nx.DiGraph()

    G.add_node("START", token=model_name, prob="START", color="lightgreen", size=4000)

    # First, create all main token nodes in sequence
    for i, pred in enumerate(predictions):
        token_id = f"t{i}"
        G.add_node(
            token_id,
            token=pred["token"],
            prob=f"{pred['probability'] * 100:.1f}%",
            color="lightblue",
            size=6000,
        )

        if i == 0:
            G.add_edge("START", token_id)
        else:
            G.add_edge(f"t{i - 1}", token_id)

    # Then add alternative nodes with a different y-position
    last_id = None
    for i, pred in enumerate(predictions):
        parent_token = "START" if i == 0 else f"t{i - 1}"

        # Add alternative token nodes slightly below main sequence
        for j, (alt_token, alt_prob) in enumerate(pred["alternatives"]):
            alt_id = f"t{i}_alt{j}"
            G.add_node(
                alt_id, token=alt_token, prob=f"{alt_prob * 100:.1f}%", color="lightgray", size=6000
            )

            # Add edge from main token to its alternatives only
            G.add_edge(parent_token, alt_id)
        
        if predictions:
            if i == len(predictions) -1:
                last_id = f"t{i}"


    G.add_node("END", token="END", prob="100%", color="red", size=6000)
    if last_id:
        G.add_edge(last_id, "END")
    else:
        G.add_edge("START", "END")


    return G


def _visualize_predictions_plt(G: nx.DiGraph, figsize=(14, 80)):
    """
    Visualize the token prediction graph with vertical layout and alternating alternatives.
    """
    plt.figure(figsize=figsize)

    # Create custom positioning for nodes
    pos = {}
    spacing_y = 5  # Vertical spacing between main tokens
    spacing_x = 5  # Horizontal spacing for alternatives

    # Position main token nodes in a vertical line
    main_nodes = [n for n in G.nodes() if "_alt" not in n]
    for i, node in enumerate(main_nodes):
        pos[node] = (0, -i * spacing_y)  # Center main tokens vertically

    # Position alternative nodes to left and right of main tokens
    for node in G.nodes():
        if "_alt" in node:
            main_token_prefix = node.split("_")[0]
            # Find the corresponding main token node
            main_token_node = None
            if main_token_prefix == 't0':
                main_token_node = 'START'
            else:
                # This logic assumes main_token_prefix is like 't<number>'
                try:
                    main_token_index = int(main_token_prefix[1:])
                    if main_token_index > 0:
                        main_token_node = f"t{main_token_index - 1}"
                    else:
                        main_token_node = "START"
                except (ValueError, IndexError):
                    main_token_node = "START" # Fallback

            alt_num = int(node.split("_alt")[1])
            if main_token_node in pos:
                # Place first alternative to left, second to right
                x_offset = -spacing_x if alt_num == 0 else spacing_x
                pos[node] = (pos[main_token_node][0] + x_offset, pos[main_token_node][1] - spacing_y/2)


    # Draw nodes
    node_colors = [G.nodes[node]["color"] for node in G.nodes()]
    node_sizes = [G.nodes[node]["size"] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

    # Draw all edges as straight lines
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, alpha=0.7)

    # Add labels with token and probability
    labels = {node: f"{G.nodes[node]['token']}\n{G.nodes[node]['prob']}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=14)

    plt.title("Token prediction.")
    plt.axis("off")

    # Adjust plot limits to ensure all nodes are visible
    margin = 8
    x_values = [x for x, y in pos.values()]
    y_values = [y for x, y in pos.values()]
    plt.xlim(min(x_values) - margin, max(x_values) + margin)
    plt.ylim(min(y_values) - margin, max(y_values) + margin)

    # plt.tight_layout()
    return plt

def visualizer(prompt: str, model_provider: str, model_name: str, max_tokens: int = 10):
    predictor = TokenPredictor(model_provider=model_provider, model_name=model_name)
    predictions = predictor.predict_tokens(prompt, max_tokens=max_tokens)
    graph = create_token_graph(model_name, predictions)
    return _visualize_predictions_plt(graph)