from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from chat_engine import ChatEngine
import traceback

app = Flask(__name__)
CORS(app)

# Initialize the chat engine (loads RAG + model on startup)
print("🚀 Starting ShopMart AI Customer Assistant...")
chat_engine = ChatEngine(model="gemma3:4b")


@app.route("/")
def index():
    """Serve the chatbot UI."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint — receives message, returns AI response."""
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        return Response(
            chat_engine.stream_chat(user_message), 
            mimetype="application/x-ndjson"
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Something went wrong. Please try again.",
            "detail": str(e)
        }), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset conversation history."""
    result = chat_engine.reset_conversation()
    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def history():
    """Get conversation history."""
    return jsonify({"history": chat_engine.get_history()})


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": chat_engine.model,
        "knowledge_base_docs": "loaded"
    })


if __name__ == "__main__":
    print("🌐 Server starting at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)