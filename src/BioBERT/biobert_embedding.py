import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd



def load_biobert(device=None):
    # If no device is provided, automatically choose GPU if available, otherwise CPU
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the tokenizer for BioBERT
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

    # Load the pretrained BioBERT model from Hugging Face
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    
    # Move the model to the selected device and set it to evaluation mode
    # eval() disables training-specific behavior such as dropout
    model = model.to(device).eval()
    return tokenizer, model, device


def embed_texts(texts, model, tokenizer, device, batch_size=32):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]

        # 1) Tokenize the batch
        encoded = tokenizer(
            batch,
            padding=True,          # pad shorter texts so all texts in the batch have equal length
            truncation=True,       # cut off texts that are longer than max_length
            max_length=512,        # maximum length: 512 tokens
            return_tensors="pt"    # return PyTorch tensors
            ).to(device)               # move tensors to CPU or GPU

        # 2)
        with torch.no_grad():  # disable gradient calculation for inference
            outputs = model(**encoded)  # forward pass through the model
        
        # CLS token = first token of last_hidden_state, shape (batch, 768)
        embeddings = outputs.last_hidden_state[:, 0, :]

        # 3) Move embeddings back to CPU and convert to NumPy array, then store in list
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)   # shape (N, 768)    


if __name__ == "__main__":

    df = pd.read_csv("data/luad/descriptions.csv", index_col=0)
    texts = df["description"].tolist()

    tokenizer, model, device = load_biobert()
    embeddings = embed_texts(texts, model, tokenizer, device)

    np.savez("data/luad/text_embeddings.npz",
         embeddings=embeddings,
         patient_ids=df.index.to_numpy())

    print(f"Saved embeddings: {embeddings.shape}")
