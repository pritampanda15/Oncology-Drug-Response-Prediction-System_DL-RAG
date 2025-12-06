# src/tools/prediction_tool.py

import os
# Fix SSL certificate issue from PyMOL
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import mlflow
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

from src.data.loader import GDSCDataLoader
from src.data.preprocessor import GeneExpressionPreprocessor, prepare_splits
from src.models.predictor import DrugResponsePredictor as NeuralModel
from src.models.trainer import Trainer, create_dataloader
from src.rag.retriever import KnowledgeBaseRetriever
from src.rag.generator import ExplanationGenerator


class DrugResponseTool:
    """
    Complete drug response prediction system with RAG-based explanations.
    Combines deep learning models with knowledge base retrieval.
    """

    def __init__(
        self,
        models_dir: str = "models",
        vector_store_dir: str = "knowledge_base/vector_store",
        use_rag: bool = True
    ):
        self.models_dir = Path(models_dir)
        self.vector_store_dir = Path(vector_store_dir) if vector_store_dir else None
        self.loaded_models = {}
        self.use_rag = use_rag

        # Initialize RAG components if enabled
        if self.use_rag and self.vector_store_dir and self.vector_store_dir.exists():
            try:
                self.retriever = KnowledgeBaseRetriever(
                    persist_directory=str(self.vector_store_dir)
                )
                self.generator = ExplanationGenerator()
                print(f"✅ RAG system initialized with {self.retriever.get_collection_stats()['total_documents']} documents")
            except Exception as e:
                print(f"⚠️  RAG initialization failed: {e}. Falling back to basic explanations.")
                self.use_rag = False
                self.retriever = None
                self.generator = None
        else:
            print("ℹ️  RAG disabled or vector store not found. Using basic explanations.")
            self.use_rag = False
            self.retriever = None
            self.generator = None


    def _load_model(self, drug_name: str) -> Dict:
        """Load a trained model and its metadata."""
        model_path = self.models_dir / f"{drug_name.lower()}_predictor.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found for {drug_name} at {model_path}. "
                f"Please train the model first using scripts/train_model.py"
            )

        checkpoint = torch.load(model_path, map_location="cpu")

        # Create model instance
        model = NeuralModel(
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout_rate=checkpoint["dropout_rate"]
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        return {
            "model": model,
            "selected_genes": checkpoint["selected_genes"],
            "input_dim": checkpoint["input_dim"],
            "drug_name": checkpoint["drug_name"],
            "best_auc": checkpoint.get("best_auc", None)
        }

    def predict_and_explain(
        self,
        drug_name: str,
        gene_expression: pd.DataFrame
    ) -> Dict:
        """
        Make prediction and generate explanation.

        Args:
            drug_name: Name of drug (e.g., "Cisplatin")
            gene_expression: DataFrame with gene expression values (1 row)

        Returns:
            Dictionary with prediction, confidence, top genes, and explanation
        """
        # Load model if not already loaded
        if drug_name not in self.loaded_models:
            self.loaded_models[drug_name] = self._load_model(drug_name)

        model_info = self.loaded_models[drug_name]
        model = model_info["model"]
        selected_genes = model_info["selected_genes"]

        # Ensure gene_expression is a DataFrame with string columns
        gene_expression = gene_expression.copy()
        gene_expression.columns = gene_expression.columns.astype(str)

        # Select only the genes used during training
        missing_genes = [g for g in selected_genes if g not in gene_expression.columns]
        if missing_genes:
            raise ValueError(
                f"Missing {len(missing_genes)} required genes. "
                f"Examples: {missing_genes[:5]}"
            )

        X = gene_expression[selected_genes].values

        # Make prediction
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            output = model(X_tensor)
            prob = torch.sigmoid(output).item()
            prediction = "Responder" if prob >= 0.5 else "Non-Responder"

        # Get top contributing genes (simple importance based on magnitude)
        gene_values = X[0]
        gene_importance = np.abs(gene_values)
        top_indices = np.argsort(gene_importance)[::-1][:10]
        top_genes = [selected_genes[i] for i in top_indices]

        # Generate explanation (placeholder for now - can integrate RAG later)
        explanation = self._generate_explanation(
            drug_name, prediction, prob, top_genes, gene_expression
        )

        return {
            "drug_name": drug_name,
            "prediction": prediction,
            "confidence": prob if prediction == "Responder" else (1 - prob),
            "top_genes": top_genes,
            "explanation": explanation
        }

    def _generate_explanation(
        self,
        drug_name: str,
        prediction: str,
        prob: float,
        top_genes: List[str],
        gene_expression: pd.DataFrame
    ) -> str:
        """Generate human-readable explanation using RAG or fallback to basic explanation."""

        model_info = self.loaded_models[drug_name]
        best_auc = model_info.get("best_auc", "unknown")

        # Use RAG if available
        if self.use_rag and self.retriever and self.generator:
            try:
                # Build query from drug name and top genes
                gene_list = ", ".join(top_genes[:5])
                query = f"{drug_name} drug response mechanism {gene_list}"

                # Retrieve relevant context
                retrieved = self.retriever.search(
                    query=query,
                    n_results=3,
                    drug_filter=drug_name.lower()
                )

                # Generate explanation using LLM
                explanation = self.generator.generate_explanation(
                    drug_name=drug_name,
                    prediction=prediction,
                    confidence=prob if prediction == "Responder" else (1 - prob),
                    top_genes=top_genes,
                    retrieved_context=retrieved
                )

                # Add model performance info
                auc_str = f"{best_auc:.4f}" if isinstance(best_auc, float) else str(best_auc)
                explanation = f"Model Performance (AUC): {auc_str}\n\n{explanation}"

                return explanation

            except Exception as e:
                print(f"⚠️  RAG explanation failed: {e}. Using fallback.")

        # Fallback to basic explanation
        auc_str = f"{best_auc:.4f}" if isinstance(best_auc, float) else str(best_auc)
        conf_str = f"{prob * 100:.1f}%"

        explanation = f"""Based on the deep learning model trained on GDSC2 data (AUC: {auc_str}):

Prediction: {prediction}
Confidence: {conf_str}

Top Contributing Genes:
"""
        for i, gene in enumerate(top_genes[:5], 1):
            expr_val = gene_expression[gene].iloc[0]
            explanation += f"  {i}. {gene} (expression: {expr_val:.2f})\n"

        explanation += f"""
The model analyzed {model_info['input_dim']} gene expression features to predict
drug response to {drug_name}. The prediction is based on patterns learned from
cancer cell line screening data.
"""
        return explanation.strip()


def _ensure_dataframe(X: any, columns: List[str] = None) -> pd.DataFrame:
    """Convert anything (DataFrame, ndarray, tensor) → DataFrame with string columns"""
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        X.columns = X.columns.astype(str)
        return X

    # Convert numpy/torch → DataFrame
    if hasattr(X, "columns"):  # already has columns (unlikely here)
        X = pd.DataFrame(X)
        X.columns = X.columns.astype(str)
        return X

    # Most common case: raw numpy array
    X = pd.DataFrame(X)

    if columns is not None:
        if len(columns) != X.shape[1]:
            raise ValueError(f"Provided {len(columns)} column names but data has {X.shape[1]} features")
        X.columns = [str(c) for c in columns]
    else:
        X.columns = [str(i) for i in range(X.shape[1])]

    return X


def train(
    drug_name: str = "Cisplatin",
    n_genes: int = 1000,           # 0 or negative → use ALL genes
    hidden_dims: list = None,
    dropout_rate: float = 0.5,
    weight_decay: float = 0.01,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.0005
):
    if hidden_dims is None:
        hidden_dims = [256, 128, 32]

    use_all_genes = (n_genes is None or n_genes <= 0 or n_genes >= 20000)

    mlflow.set_experiment("drug_response_prediction")
    run_name = f"{drug_name}_{'allgenes' if use_all_genes else n_genes}genes"
    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "drug_name": drug_name,
            "n_genes_requested": n_genes if not use_all_genes else "ALL",
            "use_all_genes": use_all_genes,
            "hidden_dims": str(hidden_dims),
            "dropout_rate": dropout_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        })

        # ==================== DATA LOADING ====================
        loader = GDSCDataLoader("data/raw")
        loader.load_expression("Cell_line_RMA_proc_basalExp.txt")
        loader.load_drug_response("GDSC2_fitted_dose_response_27Oct23.xlsx")

        X_raw, y = loader.get_dataset_for_drug(drug_name)
        X_raw.columns = X_raw.columns.astype(str)

        X_train, X_test, y_train, y_test = prepare_splits(X_raw, y)

        # ==================== PREPROCESSING ====================
        if use_all_genes:
            print(f"Training {drug_name} on ALL available genes (no selection)")
            preprocessor = GeneExpressionPreprocessor(
                n_top_genes=None,
                drug_name=drug_name,
                prioritize_known_genes=False
            )
        else:
            print(f"Training {drug_name} on top {n_genes} genes")
            preprocessor = GeneExpressionPreprocessor(
                n_top_genes=n_genes,
                drug_name=drug_name,
                prioritize_known_genes=True
            )

        # These may return DataFrame OR numpy array → we handle both
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # ==================== UNIFY TO DATAFRAME + EXTRACT GENE NAMES ====================
        # Keep original gene order from the raw data if possible
        if hasattr(preprocessor, "selected_genes") and preprocessor.selected_genes is not None:
            gene_names = [str(g) for g in preprocessor.selected_genes]
        else:
            # Fall back to original columns (in same order as input)
            gene_names = X_raw.columns.tolist()

        # Convert to DataFrame with correct column names
        X_train_processed = _ensure_dataframe(X_train_processed, columns=gene_names)
        X_test_processed  = _ensure_dataframe(X_test_processed,  columns=gene_names)

        # Final sanity check
        final_genes = X_train_processed.columns.astype(str).tolist()
        actual_input_dim = X_train_processed.shape[1]

        assert len(final_genes) == actual_input_dim, \
            f"Gene list length {len(final_genes)} != input dim {actual_input_dim}"

        print(f"Final input shape: {X_train_processed.shape}")
        print(f"Using {actual_input_dim} genes for {drug_name}")

        # ==================== DATALOADERS ====================
        train_loader = create_dataloader(X_train_processed.values, y_train, batch_size)
        test_loader  = create_dataloader(X_test_processed.values,  y_test,  batch_size, shuffle=False)

        mlflow.log_params({
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "input_dim": actual_input_dim,
            "n_genes_used": len(final_genes)
        })

        # ==================== MODEL ====================
        model = NeuralModel(
            input_dim=actual_input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        trainer = Trainer(model, learning_rate=learning_rate, weight_decay=weight_decay)

        print(f"\nModel: {actual_input_dim} → {hidden_dims} → 1")
        print(f"Device: {trainer.device}\n")

        # ==================== TRAINING LOOP ====================
        best_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            train_loss = trainer.train_epoch(train_loader)
            metrics = trainer.evaluate(test_loader)
            trainer.step_scheduler(metrics["auc"])

            mlflow.log_metrics({
                "train_loss": train_loss,
                "test_auc": metrics["auc"],
                "test_accuracy": metrics["accuracy"],
                "test_f1": metrics["f1"],
                "lr": trainer.get_lr()
            }, step=epoch)

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"Epoch {epoch+1:3d} → NEW BEST AUC: {best_auc:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(best_state)

        # ==================== SAVE MODEL (100% SAFE) ====================
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{drug_name.lower()}_predictor.pt"

        torch.save({
            "model_state": best_state,
            "selected_genes": final_genes,           # Correct order & length
            "input_dim": actual_input_dim,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
            "drug_name": drug_name,
            "best_auc": float(best_auc),
            "best_epoch": best_epoch + 1,
            "n_genes_used": len(final_genes),
            "feature_selection": not use_all_genes
        }, model_path)

        mlflow.log_artifact(model_path)
        mlflow.log_metric("final_best_auc", best_auc)

        print("\n" + "="*70)
        print(f"SUCCESS: {drug_name.upper()} trained!")
        print(f"Best AUC: {best_auc:.4f} (epoch {best_epoch+1})")
        print(f"Input genes: {len(final_genes)} → saved to {model_path}")
        print("="*70 + "\n")

        return model, preprocessor


# Alias for backward compatibility
DrugResponsePredictor = DrugResponseTool


if __name__ == "__main__":
    drugs = ["Cisplatin", "Docetaxel", "Paclitaxel", "Gemcitabine"]

    for drug in drugs:
        print(f"\n{'='*80}")
        print(f"TRAINING {drug.upper()}")
        print(f"{'='*80}\n")
        try:
            train(drug_name=drug, n_genes=1000)        # Change to 0 for full-genome models
        except Exception as e:
            print(f"FAILED {drug}: {e}")
            import traceback
            traceback.print_exc()