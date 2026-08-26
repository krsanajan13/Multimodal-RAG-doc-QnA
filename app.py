from flask import Flask, render_template, request, jsonify, send_from_directory
from retrieval.retriever import Retriever
from generation.answer_generator import generate_answer
import config
import json
import os

# Initialize Flask with specific folder paths
app = Flask(__name__, template_folder="frontend", static_folder="frontend")

# Load the retriever model using paths from config
retriever = Retriever(
    index_path=str(config.FAISS_INDEX_FILE),
    id2meta_path=str(config.ID2META_FILE)
)

@app.route("/")
def home():
    """Renders the main UI page."""
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """
    Handles the Question Answering logic:
    1. Receives query from frontend.
    2. Retrieves relevant chunks (text & images).
    3. Generates an answer using the LLM.
    4. Returns JSON with separated text and image sources.
    """
    data = request.json
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Empty question"}), 400

    # 1. Retrieve top relevant chunks (increased to 6 to ensure mix of modalities)
    retrieved = retriever.query(query, top_k=6)

    # 2. Get generated answer from LLM
    final_answer = generate_answer(query, retrieved)

    # 3. Reformat sources for the frontend UI
    text_sources = []
    image_sources = []

    for c in retrieved:
        # Handle Text and Table chunks
        if c["type"] in ["text", "table"]:
            text_sources.append({
                "chunk_id": c.get("chunk_id", "N/A"),
                "page": c.get("page", "N/A"),
                "text": c.get("text", "No text available.")
            })
        
        # Handle Image chunks
        elif c["type"] == "image":
            image_sources.append({
                "chunk_id": c.get("chunk_id", "Figure"),
                "page": c.get("page", "N/A"),
                # Ensure the path corresponds to the route below
                "image": c.get("image_path", "").replace("\\", "/") 
            })

    # Return structured JSON
    return jsonify({
        "answer": final_answer.get("answer", "No answer generated."),
        "sources_text": text_sources,
        "sources_image": image_sources
    })

# --- Helper Route to Serve Images ---
# This is required because images are likely stored in 'data/' or 'processed/', 
# which are outside the 'frontend/' static folder.
@app.route('/data/<path:filename>')
def serve_data_images(filename):
    # Adjust 'data' to match the root folder where your images are stored relative to app.py
    return send_from_directory('data', filename)

if __name__ == "__main__":
    app.run(debug=True, port=3000)