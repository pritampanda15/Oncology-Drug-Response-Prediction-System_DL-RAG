"""
src/data/preprocessor.py

Feature selection using biological priors combined with variance filtering.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml


class GeneExpressionPreprocessor:
    
    def __init__(
        self, 
        n_top_genes: int = 2000,
        drug_name: str = None,
        gene_sets_path: str = "config/gene_sets.yaml",
        prioritize_known_genes: bool = True
    ):
        self.n_top_genes = n_top_genes
        self.drug_name = drug_name
        self.gene_sets_path = Path(gene_sets_path)
        self.prioritize_known_genes = prioritize_known_genes
        self.scaler = StandardScaler()
        self.selected_genes = None
        self.known_genes = []
        self.known_genes_found = []
    
    def _load_known_genes(self) -> list:
        """Load drug-specific genes from config."""
        if not self.gene_sets_path.exists():
            print(f"Warning: {self.gene_sets_path} not found, using variance only")
            return []
        
        with open(self.gene_sets_path, "r") as f:
            gene_sets = yaml.safe_load(f)
        
        drug_key = self.drug_name.lower() if self.drug_name else None
        if drug_key not in gene_sets:
            print(f"Warning: No gene set for {self.drug_name}, using variance only")
            return []
        
        known_genes = []
        for pathway, genes in gene_sets[drug_key].items():
            known_genes.extend(genes)
        
        return list(set(known_genes))
    
    def fit(self, X: pd.DataFrame):
        X.columns = X.columns.astype(str)

        # Handle case where we want ALL genes (no feature selection)
        if self.n_top_genes is None:
            print(f"Using ALL {X.shape[1]} genes (no feature selection)")
            self.selected_genes = X.columns.tolist()
            self.known_genes = []
            self.known_genes_found = []
            X_selected = X[self.selected_genes]
            self.scaler.fit(X_selected)
            return

        # 1. Load and find known genes
        self.known_genes = self._load_known_genes()  # FIX: Save the result!
        self.known_genes_found = [g for g in self.known_genes if g in X.columns]

        # 2. Calculate remaining slots for high-variance genes
        remaining_slots = self.n_top_genes - len(self.known_genes_found)

        # 3. Select high-variance genes to fill the remaining slots
        if remaining_slots > 0:
            other_genes = [g for g in X.columns if g not in self.known_genes_found]
            variance = X[other_genes].var(axis=0).sort_values(ascending=False)
            top_variance_genes = variance.head(remaining_slots).index.tolist()
        else:
            top_variance_genes = []

        # 4. CRITICAL: Save the final list, which should be exactly n_top_genes features
        self.selected_genes = self.known_genes_found + top_variance_genes

        print(f"Selected {len(self.selected_genes)} genes:")
        print(f"  - {len(self.known_genes_found)} known drug-response genes")
        print(f"  - {len(top_variance_genes)} high-variance genes")

        # 5. Fit the scaler only on the selected genes
        X_selected = X[self.selected_genes]
        self.scaler.fit(X_selected)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        X.columns = X.columns.astype(str)
        X_selected = X[self.selected_genes]
        return self.scaler.transform(X_selected)
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
    def get_gene_categories(self) -> dict:
        """Return which genes are known vs variance-selected."""
        return {
            "known_genes": self.known_genes_found,
            "variance_genes": [g for g in self.selected_genes if g not in self.known_genes_found]
        }


def prepare_splits(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )