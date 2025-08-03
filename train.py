import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

from config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_NAME
from dataset import get_dataloaders
from models import get_model


def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    probs = []
    loss_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item()

            preds = torch.argmax(outputs, dim=1)
            prob = torch.softmax(outputs, dim=1)[:, 1] 

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            probs.extend(prob.detach().cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": specificity,
        "auc": auc
    }


def save_metrics(model_name, metrics, csv_path="results.csv"):
    
    row = {
        "Model": model_name,
        "Accuracy (%)": round(metrics["accuracy"] * 100, 1),
        "Precision": round(metrics["precision"], 3),
        "Recall (Sensitivity)": round(metrics["recall"], 3),
        "F1-Score": round(metrics["f1_score"], 3),
        "Specificity": round(metrics["specificity"], 3),
        "AUC": round(metrics["auc"], 3)
    }

    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    model = get_model(MODEL_NAME).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f}")

    print("Evaluating on validation set.")
    val_metrics = evaluate(model, val_loader, criterion)

    print("Evaluating on test set.")
    test_metrics = evaluate(model, test_loader, criterion)

    print("Saving test metrics to CSV.")
    save_metrics(MODEL_NAME, test_metrics)
