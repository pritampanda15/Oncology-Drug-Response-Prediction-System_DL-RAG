"""
src/data/loader.py

Data loading utilities for GDSC2 drug response and gene expression data.
Handles ID matching between expression (DATA.XXXXX) and response (XXXXX) formats.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


class GDSCDataLoader:
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.expression_df: Optional[pd.DataFrame] = None
        self.response_df: Optional[pd.DataFrame] = None
        self.expression_cosmic_ids: list = []
    
    def load_drug_response(
        self, 
        filename: str = "GDSC2_fitted_dose_response_27Oct23.xlsx"
    ) -> pd.DataFrame:
        """
        Load GDSC2 drug response data.
        """
        filepath = self.data_dir / filename
        self.response_df = pd.read_excel(filepath)
        
        print(f"Loaded drug response: {len(self.response_df)} measurements")
        print(f"Unique drugs: {self.response_df['DRUG_NAME'].nunique()}")
        print(f"Unique cell lines: {self.response_df['COSMIC_ID'].nunique()}")
        
        return self.response_df
    
    def load_expression(
        self, 
        filename: str = "Cell_line_RMA_proc_basalExp.txt"
    ) -> pd.DataFrame:
        """
        Load RMA normalized gene expression matrix.
        Format: genes as rows, cell lines as columns (DATA.COSMIC_ID).
        """
        filepath = self.data_dir / filename

        raw = pd.read_csv(filepath, sep="\t")

        # Filter out rows with invalid gene names (NaN, empty, or 'nan' string)
        raw = raw[raw["GENE_SYMBOLS"].notna()]
        raw = raw[raw["GENE_SYMBOLS"] != ""]
        raw = raw[raw["GENE_SYMBOLS"].astype(str) != "nan"]

        raw = raw.set_index("GENE_SYMBOLS")
        raw = raw.drop(columns=["GENE_title"], errors="ignore")
        
        new_columns = {}
        for col in raw.columns:
            if col.startswith("DATA."):
                id_part = col.replace("DATA.", "")
                if "." in id_part:
                    id_part = id_part.split(".")[0]
                try:
                    cosmic_id = int(id_part)
                    new_columns[col] = cosmic_id
                except ValueError:
                    continue
        
        raw = raw.rename(columns=new_columns)
        self.expression_df = raw
        self.expression_cosmic_ids = list(raw.columns)
        
        print(f"Loaded expression: {self.expression_df.shape}")
        print(f"Genes: {self.expression_df.shape[0]}")
        print(f"Cell lines: {self.expression_df.shape[1]}")
        
        return self.expression_df
    
    def get_available_drugs(self) -> list:
        """Return sorted list of unique drug names."""
        if self.response_df is None:
            raise ValueError("Load drug response data first")
        return sorted(self.response_df["DRUG_NAME"].unique().tolist())
    
    def get_dataset_for_drug(
        self, 
        drug_name: str,
        binarize: bool = True
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Merge expression and drug response for a specific drug.
        
        Args:
            drug_name: Name of drug (e.g., "Cisplatin")
            binarize: If True, convert LN_IC50 to binary label
            
        Returns:
            X: Gene expression DataFrame (samples x genes)
            y: Target Series (binary label or continuous IC50)
        """
        if self.response_df is None or self.expression_df is None:
            raise ValueError("Load both expression and drug response data first")
        
        drug_data = self.response_df[
            self.response_df["DRUG_NAME"] == drug_name
        ].copy()
        
        if len(drug_data) == 0:
            available = self.get_available_drugs()[:10]
            raise ValueError(f"Drug '{drug_name}' not found. Examples: {available}")
        
        drug_cosmic_ids = set(drug_data["COSMIC_ID"].values)
        expression_cosmic_ids = set(self.expression_cosmic_ids)
        common_ids = list(drug_cosmic_ids & expression_cosmic_ids)
        
        if len(common_ids) == 0:
            raise ValueError("No overlapping cell lines between datasets")
        
        X = self.expression_df[common_ids].T
        X.index.name = "COSMIC_ID"
        
        drug_data_filtered = drug_data[
            drug_data["COSMIC_ID"].isin(common_ids)
        ].drop_duplicates(subset=["COSMIC_ID"], keep="first")
        drug_data_indexed = drug_data_filtered.set_index("COSMIC_ID")
        
        y = drug_data_indexed.loc[X.index, "LN_IC50"]
        
        if binarize:
            median_ic50 = y.median()
            y = (y < median_ic50).astype(int)
            y.name = "label"
        
        print(f"Drug: {drug_name}")
        print(f"Samples: {len(y)}")
        if binarize:
            print(f"Responders (label=1): {y.sum()}")
            print(f"Non-responders (label=0): {(y == 0).sum()}")
        
        return X, y
