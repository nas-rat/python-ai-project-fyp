from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

@app.route('/process', methods=['POST'])
def process():
    try:
        # Receive JSON data from the request
        data = request.json  
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400  # Bad request

        # Tokenize the text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        # Get embeddings from the model
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embedding
        cls_embedding_np = cls_embedding.detach().numpy()  # Convert to numpy

        # Return the CLS embedding as JSON response
        return jsonify({"cls_embedding": cls_embedding_np.tolist()}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500  # Internal server error

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
