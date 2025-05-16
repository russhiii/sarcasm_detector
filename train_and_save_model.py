#train_and_save_model.py
import torch
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm

# Check device: GPU if available else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataset from .jsonl file (each line is a JSON object)
with open("dataset.jsonl", "r") as file:
    data = [json.loads(line) for line in file]

# Create DataFrame with required columns
df = pd.DataFrame(data)[['headline', 'is_sarcastic']]
df.columns = ['text', 'label']

# Train-validation split (stratified to keep label balance)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(),
    test_size=0.2, random_state=42, stratify=df['label']
)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize with padding and truncation
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)

# Custom Dataset class for PyTorch
class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = SarcasmDataset(train_encodings, train_labels)
val_dataset = SarcasmDataset(val_encodings, val_labels)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pretrained BERT model for sequence classification (2 classes)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)

# Training settings
num_epochs = 5
num_training_steps = num_epochs * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

# Scheduler for learning rate warmup and decay
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Training loop with validation
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        total_train_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())
        
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1} training loss: {avg_train_loss:.4f}")
    
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            
            total_val_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct / total
    print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}, accuracy: {val_accuracy:.4f}")

# Save the fine-tuned model and tokenizer for later use
model.save_pretrained("saved_model_dir")
tokenizer.save_pretrained("saved_model_dir")

print("✅ Model and tokenizer saved in 'saved_model_dir'")
