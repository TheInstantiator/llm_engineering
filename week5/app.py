# Gradio is a Python library that allows you to create web-based user interfaces for your machine learning models (or any Python function) very quickly.
import gradio as gr
from dotenv import load_dotenv

# We import the 'answer_question' function from our pro implementation. 
# This is the "brain" that will handle the logic of searching and answering.
from pro_implementation.answer import answer_question

load_dotenv(override=True)


# Helper function to prettify the retrieved context for display in the UI.
# It formats the text as HTML, highlighting the source of the document.
def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        # Display the source filename clearly
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        # Display the actual content content
        result += doc.page_content + "\n\n"
    return result


# The main chat logic function called by the UI.
# 'history' contains the list of messages in the conversation so far: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
def chat(history):
    # Get the latest user message (the last one added to the list)
    last_message = history[-1]["content"]
    # Get all previous messages to serve as conversation context
    prior = history[:-1]
    
    # Call our backbone RAG function to get the answer and the source documents (context)
    answer, context = answer_question(last_message, prior)
    
    # Append the AI's response to the history so the chatbot displays it
    history.append({"role": "assistant", "content": answer})
    
    # Return the updated history (for the chat window) and the formatted context (for the side panel)
    return history, format_context(context)


def main():
    # A simple helper to immediately add the user's message to the chat window 
    # before we wait for the AI to generate a response.
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    # Define the visual theme of the app (fonts, colors).
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    # gr.Blocks is the low-level API that lets us design custom layouts (rows and columns).
    with gr.Blocks(title="Insurellm Expert Assistant", theme=theme) as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        # Create a layout with two columns side-by-side.
        with gr.Row():
            # LEFT COLUMN: The Chat Interface
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", height=600, type="messages", show_copy_button=True
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            # RIGHT COLUMN: The Retrieved Documents Display
            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        # Connect the components:
        # 1. When the user asserts Enter in the text box (message.submit), run 'put_message_in_chatbot'.
        #    - Inputs: The current text message and history.
        #    - Outputs: Clears the text box ("") and updates the chatbot history with the user's message.
        # 2. THEN (.then), run the 'chat' function.
        #    - Inputs: The updated history (which now includes the user's message).
        #    - Outputs: Updates the 'chatbot' with the AI answer AND updates 'context_markdown' with the docs.
        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    # Launch the web server. 'inbrowser=True' tries to open a new tab automatically.
    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
