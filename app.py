from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# ✅ Load model and tokenizer from the saved directory
MODEL_DIR = "saved_model_dir"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # CPU is automatically used if GPU not available

# Predict function
def predict_sarcasm(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Sarcastic " if prediction == 1 else "Not Sarcastic "

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form['headline']  # ✅ Matches name="headline" in your HTML
    result = predict_sarcasm(text)
    return render_template("result.html", headline=text, prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
