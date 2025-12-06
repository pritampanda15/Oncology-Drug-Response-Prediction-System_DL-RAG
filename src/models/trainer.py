"""
src/models/trainer.py

Training loop with weight decay and learning rate scheduling.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np


class Trainer:
    
    def __init__(
        self, 
        model, 
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch).squeeze()
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch).squeeze()
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())
        
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "auc": roc_auc_score(all_labels, all_probs),
            "f1": f1_score(all_labels, all_preds)
        }
        
        return metrics
    
    def step_scheduler(self, metric: float):
        self.scheduler.step(metric)
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def create_dataloader(X: np.ndarray, y, batch_size: int = 32, shuffle: bool = True):
    if hasattr(y, "values"):
        y = y.values
    
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)