import json, random, os, re, time, gc
import torch
import pandas as pd
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAME_FILE     = "datasets\data.json"  
CHITCHAT_FILE = "datasets/intent/chitchat_3150.json"     
MANUAL_RECOMM_FILE = "datasets/intent/recommendation.json"       
OUT_FILE      = "datasets/intent/intent_dataset.jsonl"         
 

with open(GAME_FILE, "r", encoding="utf-8") as f:
    games = json.load(f)

with open(CHITCHAT_FILE, "r", encoding="utf-8") as f:
    chitchat = json.load(f)

def make_prompts(game):
    title   = game["title"]
    tags    = game.get("tags", [])[:3]
    tag_str = ", ".join(tags)
    rating  = game.get("rating", "Positive").lower()

    raw = [
        f"Can you recommend a game like {title}?",
        f"I'm looking for a {tag_str} game. Any ideas?" if tag_str else "",
        f"What do you think of {title}?",
        f"Know any games similar to {title}?",
        f"Any good {random.choice(tags)} games?" if tags else "",
        f"Suggest a popular game with {tag_str}.",
        f"Is {title} worth playing?",
        f"Recommend a {rating} game similar to {title}."
    ]
    return [s for s in raw if s]

recommend = []
random.shuffle(games)

for g in games:
    prompts = make_prompts(g)
    for sent in random.sample(prompts, k=min(3, len(prompts))):
        recommend.append({"text": sent, "label": "recommendation"})
    if len(recommend) >= 3000:
        break

with open(MANUAL_RECOMM_FILE, "r", encoding="utf-8") as f:
    manual_recommend = json.load(f)

dataset = manual_recommend + recommend + chitchat
total_recommend = len(recommend) + len(manual_recommend)
random.shuffle(dataset)
print(f"Total dataset size: {len(dataset)} "
      f"(recommendation {(total_recommend)}, chitchat {len(chitchat)})")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Dataset written to {OUT_FILE}")


# Loading data
with open("datasets/intent/intent_dataset.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f]

df = pd.DataFrame(data)
print("Loaded samples:", len(df))
print(df["label"].value_counts())

# Label encoding & word segmentation
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Building Dataset and DataLoader
class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label_id"].tolist(), test_size=0.2, random_state=42
)

train_dataset = IntentDataset(train_texts, train_labels)
val_dataset = IntentDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# Loading Model & Optimizer
MODEL_DIR = "intent_model"

# Determine whether the model has been trained
if os.path.exists(MODEL_DIR):
    print("Found existing model. Loading from disk...")
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    with open(os.path.join(MODEL_DIR, "label_encoder.json"), "r", encoding="utf-8") as f:
        label_info = json.load(f)
        id2label = {i: label for i, label in enumerate(label_info["classes"])}
else:
    print("No existing model found. Starting training from scratch...")
    
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    model.to(device)

    # Training Loop
    epochs = 3
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        train_preds = []
        train_labels_true = []

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in train_loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_true.extend(batch["labels"].cpu().numpy())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels_true, train_preds)

        model.eval()
        total_val_loss = 0
        val_preds = []
        val_labels_true = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_true.extend(batch["labels"].cpu().numpy())
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels_true, val_preds)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss:     {avg_train_loss:.4f}")
        print(f"  Train Accuracy: {train_acc * 100:.2f}%")
        print(f"  Val Loss:       {avg_val_loss:.4f}")
        print(f"  Val Accuracy:   {val_acc * 100:.2f}%\n")

    # Save Model 
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    with open(os.path.join(MODEL_DIR, "label_encoder.json"), "w") as f:
        json.dump({"classes": list(label_encoder.classes_)}, f)
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    print("Model and tokenizer saved.")


# Load label mapping
with open("intent_model/label_encoder.json", "r", encoding="utf-8") as f:
    label_info = json.load(f)
    id2label = {i: label for i, label in enumerate(label_info["classes"])}

# Define the inference function
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]

# Test
print(predict_intent("can you recommend me games"))     
print(predict_intent("I like your recommendation yesterday"))  

# Chithat reply generation
# Model & Tokenizer 
phi_name   = "microsoft/phi-2"
chat_tok   = AutoTokenizer.from_pretrained(phi_name, use_fast=True)
chat_tok.pad_token = chat_tok.eos_token
chat_model = AutoModelForCausalLM.from_pretrained(
    phi_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
chat_model.config.pad_token_id = chat_tok.eos_token_id
chat_model.eval()

persona = (
    "You are GamerBuddy, a friendly assistant who likes talking about games, "
    "food, and daily life. Always answer the user's last question directly and briefly."
)

BAD_WORDS      = ["fuck", "shit"]
MAX_HISTORY    = 3
dialog_history = []         

def sanitize(text: str) -> str:
    pattern = "|".join(map(re.escape, BAD_WORDS))
    return re.sub(pattern, "[censored]", text, flags=re.I)

def generate_chitchat_reply(user_input: str, max_new_tokens: int = 60) -> str:
    global dialog_history
    user_input = sanitize(user_input)

    # Push the latest user's speech into history
    dialog_history.append(f"User: {user_input}")
    dialog_history = dialog_history[-MAX_HISTORY:]

    # Reassemble the complete prompt word
    prompt = f"{persona}\n" + "\n".join(dialog_history) + "\nAssistant:"
    inputs = chat_tok(prompt, return_tensors="pt", truncation=True,
                      max_length=512).to(chat_model.device)

    with torch.no_grad():
        out_ids = chat_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.95,
            eos_token_id=chat_tok.eos_token_id,
            pad_token_id=chat_tok.pad_token_id,
        
        )

    
    gen_text = chat_tok.decode(out_ids[0][inputs.input_ids.shape[1]:],
                               skip_special_tokens=True).lstrip()
    reply = gen_text.split("\n")[0].strip()     
    dialog_history.append(f"Assistant: {reply}")  
    return reply


print(chat_model.device)              
print(next(chat_model.parameters()).device)

print(generate_chitchat_reply("What time do you go to bed?"))
print(generate_chitchat_reply("Is the burger delicious?"))


# export functions so other files can import
__all__ = ["predict_intent", "generate_chitchat_reply"]