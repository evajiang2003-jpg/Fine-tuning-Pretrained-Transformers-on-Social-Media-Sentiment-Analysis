import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, AdamW
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import time
import seaborn as sns
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, TaskType
from collections import Counter
import re
from nltk.corpus import stopwords

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

print("Transformers Version:", transformers.__version__)

import kagglehub

# Download latest version
path = kagglehub.dataset_download("kazanova/sentiment140")

print("Path to dataset files:", path)

import os

data_dir = "/kaggle/input/sentiment140"
print(os.listdir(data_dir))

file_path = os.path.join(data_dir, "training.1600000.processed.noemoticon.csv")
columns = ["sentiment", "ids", "date", "query", "user", "text"]
df = pd.read_csv(file_path, encoding='latin-1', names=columns)

# 0=negative, 1=positive
df = df[df['sentiment'].isin([0, 4])]  
df["label"] = df["sentiment"].apply(lambda x: 0 if x == 0 else 1)

print(df["label"].value_counts())

# Stratified sampling, total of 80,000
n_total = 80000
n_neg = n_total // 2
n_pos = n_total - n_neg

df_negative = df[df['label']==0].sample(n=n_neg, random_state=42)
df_positive = df[df['label']==1].sample(n=n_pos, random_state=42)

df_sampled = pd.concat([df_negative, df_positive]).sample(frac=1, random_state=42)  


print("Sampled label distribution:")
print(df_sampled["label"].value_counts())

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_sampled["text"].tolist(),
    df_sampled["label"].tolist(),
    test_size=0.1,
    random_state=42,
    stratify=df_sampled["label"]
)

print("Train Texts:")
print(train_texts[:5])
print("Train Labels:")
print(train_labels[:5])

print("Val Texts:")
print(val_texts[:5])
print("ValLabels:")
print(val_labels[:5])

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased",
    cache_dir="./distilbert-cache",
    force_download=False
)

# Encoding
train_encodings = tokenizer(
    train_texts,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
val_encodings = tokenizer(
    val_texts,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

sample_text = train_texts[0]
encoded = tokenizer(sample_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
print(f"Sample Text：{sample_text}")
print(f"Encoded Lenth：{len(encoded['input_ids'][0])}")

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Dataset & Dataloader
train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    torch.tensor(train_labels)
)
val_dataset = TensorDataset(
    val_encodings['input_ids'],
    val_encodings['attention_mask'],
    torch.tensor(val_labels)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialization & Optimization
full_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # 二分类
)
full_model.to(device)
optimizer = AdamW(full_model.parameters(), lr=2e-5)

# Training Loop
epochs = 2
train_loss_history, val_loss_history = [], []
epoch_times = []

print("\n Full Fine-tuning \n")
start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    print(f"\nEpoch {epoch+1}/{epochs}")

    # === Training ===
    full_model.train()
    total_train_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = full_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # === Validation ===
    full_model.eval()
    total_val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = full_model(input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    print(f"Epoch {epoch+1} Time: {epoch_time:.2f} sec")
    print(f"Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation | Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print("-" * 70)

total_time = time.time() - start_time

# Summary
print("\n=== Final Summary (Full Fine-tuning) ===")
print(f"Total Training Time: {total_time/60:.2f} min")
print(f"Average Time per Epoch: {np.mean(epoch_times):.2f} sec")
print(f"Average Training Loss: {np.mean(train_loss_history):.4f}")
print(f"Average Validation Loss: {np.mean(val_loss_history):.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["negative", "positive"]))

# Confusion Matrix
cm_full = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_full, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred:Neg", "Pred:Pos"],
            yticklabels=["True:Neg", "True:Pos"])
plt.title("Confusion Matrix (Full Fine-tuning)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Dataset & Dataloader
train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    torch.tensor(train_labels)
)
val_dataset = TensorDataset(
    val_encodings['input_ids'],
    val_encodings['attention_mask'],
    torch.tensor(val_labels)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialization
base_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]  
)

lora_model = get_peft_model(base_model, lora_config)
lora_model.to(device)

optimizer = AdamW(lora_model.parameters(), lr=2e-5)

# Training Loop
epochs = 2
train_loss_history, val_loss_history = [], []
epoch_times = []

print("\n LoRA Fine-tuning \n")
start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    print(f"\nEpoch {epoch+1}/{epochs}")

    lora_model.train()
    total_train_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = lora_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    lora_model.eval()
    total_val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = lora_model(input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    print(f"Epoch {epoch+1} Time: {epoch_time:.2f} sec")
    print(f"Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation | Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print("-" * 70)

total_time = time.time() - start_time

# Summary
print("\n=== Final Summary (LoRA Fine-tuning) ===")
print(f"Total Training Time: {total_time/60:.2f} min")
print(f"Average Time per Epoch: {np.mean(epoch_times):.2f} sec")
print(f"Average Training Loss: {np.mean(train_loss_history):.4f}")
print(f"Average Validation Loss: {np.mean(val_loss_history):.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["negative", "positive"]))

# Confusion Matrix
cm_lora = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_lora, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred:Neg", "Pred:Pos"],
            yticklabels=["True:Neg", "True:Pos"])
plt.title("Confusion Matrix (LoRA Fine-tuning)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Dataset & Dataloader
train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    torch.tensor(train_labels)
)
val_dataset = TensorDataset(
    val_encodings['input_ids'],
    val_encodings['attention_mask'],
    torch.tensor(val_labels)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialization
partial_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2 
)

# Freeze
for param in partial_model.distilbert.parameters():
    param.requires_grad = False

optimizer = AdamW(partial_model.classifier.parameters(), lr=2e-4)
partial_model.to(device)

epochs = 2
train_loss_history, val_loss_history = [], []
epoch_times = []

print("\n Classifier-only Fine-tuning \n")
start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    print(f"\nEpoch {epoch+1}/{epochs}")

    partial_model.train()
    total_train_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = partial_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    partial_model.eval()
    total_val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = partial_model(input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    print(f"Epoch {epoch+1} Time: {epoch_time:.2f} sec")
    print(f"Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation | Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print("-" * 70)

total_time = time.time() - start_time

# Summary
print("\n=== Final Summary (Classifier-only Fine-tuning) ===")
print(f"Total Training Time: {total_time/60:.2f} min")
print(f"Average Time per Epoch: {np.mean(epoch_times):.2f} sec")
print(f"Average Training Loss: {np.mean(train_loss_history):.4f}")
print(f"Average Validation Loss: {np.mean(val_loss_history):.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["negative", "positive"]))

# Confusion Matrix
cm_partial = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_partial, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred:Neg", "Pred:Pos"],
            yticklabels=["True:Neg", "True:Pos"])
plt.title("Confusion Matrix (Classifier-only Fine-tuning)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

cms = [cm_full, cm_lora, cm_partial]
titles = ["Full Fine-tuning", "LoRA Fine-tuning", "Classifier-only Fine-tuning"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

vmax = max(cm.max() for cm in cms)

for ax, cm, title in zip(axes, cms, titles):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred:Neg", "Pred:Pos"],
                yticklabels=["True:Neg", "True:Pos"],
                cbar=False,
                vmin=0, vmax=vmax,
                ax=ax)
    ax.set_title(f"Confusion Matrix ({title})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True" if ax == axes[0] else "")

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=vmax))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label="Number of samples")

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

stop_words = set(stopwords.words('english'))

def analyze_errors_with_text_features(model, dataloader, texts, device, 
                                      label_names=None, top_n=10, max_words=20,
                                      ax=None, title="Error Text Length Distribution"):
    model.eval()
    all_preds, all_labels, all_texts = [], [], []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing errors")):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        start_idx = batch_idx * dataloader.batch_size
        end_idx = start_idx + len(labels)
        all_texts.extend(texts[start_idx:end_idx])

    df_results = pd.DataFrame({
        "text": all_texts,
        "true_label": all_labels,
        "pred_label": all_preds
    })
    df_errors = df_results[df_results["true_label"] != df_results["pred_label"]]
    top_errors = df_errors.head(top_n)

    error_counts = df_errors["true_label"].value_counts()
    total_counts = df_results["true_label"].value_counts()
    error_rate_by_class = (error_counts / total_counts).fillna(0)

    # Length
    df_errors['text_length'] = df_errors['text'].apply(lambda x: len(x.split()))
    length_counts = df_errors['text_length'].value_counts().sort_index()
    
    x = length_counts.index
    y = length_counts.values
    
    if ax is None:
        plt.figure(figsize=(8,4))
        ax = plt.gca()
    
    ax.bar(x, y, color='skyblue', edgecolor='none')
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Frequency") 
    ax.set_title(title)
    ax.tick_params(axis='x')


    # Top-10 Words
    words = []
    for text in df_errors['text']:
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        words.extend([w for w in text_clean.split() if w not in stop_words])
    word_counts = Counter(words)
    print(f"Top {max_words} frequent words in error texts:")
    for w, c in word_counts.most_common(max_words):
        print(f'"{w}": {c}')

    print("\n=== Example error texts for top frequent words with true labels ===")
    example_dict = {}
    
    label_map = {0: "negative", 1: "positive"} 
    
    for w, _ in word_counts.most_common(5):  
        examples = df_errors[df_errors['text'].str.contains(rf'\b{w}\b', case=False, na=False)][['text','true_label']].head(3)
        example_dict[w] = examples
    
        print(f'\nWord: "{w}"')
        for i, row in examples.iterrows():
            text_snip = row['text'][:150] + ("..." if len(row['text']) > 150 else "")
            label_str = label_map.get(row['true_label'], str(row['true_label']))
            print(f'  {i+1}. [{label_str}] {text_snip}')


    return top_errors, df_errors

fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True) 

models = [full_model, lora_model, partial_model]
dataloaders = [val_loader, val_loader, val_loader]
texts_list = [val_texts, val_texts, val_texts]  
titles = ["Full Fine-tuning", "LoRA Fine-tuning", "Classifier-only Fine-tuning"]

all_top_errors, all_df_errors = [], []

for i, (model, dataloader, texts) in enumerate(zip(models, dataloaders, texts_list)):
    top_errors, df_errors = analyze_errors_with_text_features(
        model, dataloader, texts, device,
        top_n=10, max_words=10,
        ax=axs[i],
        title=f"Error Text Length ({titles[i]})"
    )
    all_top_errors.append(top_errors)
    all_df_errors.append(df_errors)

plt.tight_layout()
plt.show()

