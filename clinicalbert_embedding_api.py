from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Select device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()
model.to(device)

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Tokenize input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        # Move inputs to device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract CLS embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = cls_embedding.cpu().numpy()

        return jsonify({"cls_embedding": cls_embedding.tolist()}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
