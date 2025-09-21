import traceback
from flask import Flask, jsonify
import vertexai
from vertexai.preview.language_models import ChatModel

# --- Configuration ---
GCP_PROJECT = "data-compound-456204-f4"
VERTEX_LOCATION = "us-central1"
VERTEX_MODEL = "chat-bison@001"

# --- Flask App ---
app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test_vertex_ai():
    """
    Initializes the Vertex AI client and sends a prompt when a request is received.
    """
    try:
        # Initialize the connection inside the request function
        print("Initializing Vertex AI client...")
        vertexai.init(project=GCP_PROJECT, location=VERTEX_LOCATION)
        
        print(f"Loading model '{VERTEX_MODEL}'...")
        model = ChatModel.from_pretrained(VERTEX_MODEL)
        
        # Start a chat and send the prompt
        chat = model.start_chat()
        prompt = "Hello, please introduce yourself in one sentence."
        print(f"Sending prompt: '{prompt}'")
        
        response = chat.send_message(prompt)
        print(f"Received response: '{response.text}'")
        
        return jsonify({"status": "success", "response": response.text})
    
    except Exception as e:
        # If anything goes wrong, return a detailed error
        error_traceback = traceback.format_exc()
        print(f"!!! AN ERROR OCCURRED !!!\n{error_traceback}")
        
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": error_traceback
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)