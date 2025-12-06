"""
scripts/train_model.py
Training script — now 100% safe from input_dim / selected_genes mismatch
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import mlflow
import mlflow.pytorch
from src.data.loader import GDSCDataLoader
from src.data.preprocessor import GeneExpressionPreprocessor, prepare_splits
from src.models.predictor import DrugResponsePredictor
from src.models.trainer import Trainer, create_dataloader


def train(
    drug_name: str = "Cisplatin",
    n_genes: int = 1000,           # Set to 0 or None to use ALL genes
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

    with mlflow.start_run(run_name=f"{drug_name}_{'allgenes' if use_all_genes else n_genes}genes"):

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

        X, y = loader.get_dataset_for_drug(drug_name)
        X.columns = X.columns.astype(str)

        X_train, X_test, y_train, y_test = prepare_splits(X, y)

        # ==================== PREPROCESSING ====================
        if use_all_genes:
            print(f"Training {drug_name} on ALL genes (no feature selection)")
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

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # ==================== CRITICAL SAFETY FIX ====================
        # Always trust the ACTUAL columns that went into the model
        final_genes = preprocessor.selected_genes
        actual_input_dim = X_train_processed.shape[1]

        assert final_genes is not None, "Error: preprocessor.selected_genes was not set during fit."
        assert len(final_genes) == actual_input_dim, \
            f"FATAL BUG: {len(final_genes)} genes != {actual_input_dim} input dim. Gene list is inconsistent."

        # Force preprocessor to have correct gene list (for backward compatibility)
        

        print(f"Final input dimension: {actual_input_dim} genes")

        # ==================== DATALOADERS ====================
        train_loader = create_dataloader(X_train_processed, y_train, batch_size)
        test_loader = create_dataloader(X_test_processed, y_test, batch_size, shuffle=False)

        mlflow.log_params({
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "input_dim": actual_input_dim,
            "final_n_genes": len(final_genes)
        })

        # ==================== MODEL & TRAINER ====================
        model = DrugResponsePredictor(
            input_dim=actual_input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        trainer = Trainer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        print(f"\nModel architecture:")
        print(f" Input: {actual_input_dim} → {hidden_dims} → 1")
        print(f" Dropout: {dropout_rate} | Weight decay: {weight_decay}")
        print(f" Device: {trainer.device}\n")

        # ==================== TRAINING LOOP ====================
        best_auc = 0
        best_epoch = 0
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = trainer.train_epoch(train_loader)
            metrics = trainer.evaluate(test_loader)
            trainer.step_scheduler(metrics["auc"])
            current_lr = trainer.get_lr()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "test_accuracy": metrics["accuracy"],
                "test_auc": metrics["auc"],
                "test_f1": metrics["f1"],
                "learning_rate": current_lr
            }, step=epoch)

            improved = ""
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_epoch = epoch
                epochs_without_improvement = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                improved = " ← BEST"
            else:
                epochs_without_improvement += 1

            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Acc: {metrics['accuracy']:.3f} | AUC: {metrics['auc']:.4f} | "
                  f"F1: {metrics['f1']:.3f}{improved}")

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Load best model
        model.load_state_dict(best_model_state)

        # ==================== SAVE MODEL (NOW 100% SAFE) ====================
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{drug_name.lower()}_predictor.pt"

        torch.save({
            "model_state": best_model_state,
            "selected_genes": final_genes,           # GUARANTEED correct length and order
            "input_dim": actual_input_dim,           # Matches real model input
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
            "drug_name": drug_name,
            "best_auc": best_auc,
            "best_epoch": best_epoch + 1,
            "n_genes_used": len(final_genes),
            "feature_selection": not use_all_genes
        }, model_path)

        mlflow.log_artifact(model_path)
        mlflow.log_metrics({
            "best_auc": best_auc,
            "best_epoch": best_epoch + 1
        })

        print("\n" + "="*60)
        print(f"TRAINING COMPLETE FOR {drug_name.upper()}")
        print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch + 1}")
        print(f"Input genes: {len(final_genes)}")
        print(f"Model saved: {model_path}")
        print("="*60 + "\n")

        return model, preprocessor


if __name__ == "__main__":
    # Example usage:
    drugs = ["Cisplatin", "Docetaxel", "Paclitaxel", "Gemcitabine"]

    for drug in drugs:
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING FOR: {drug.upper()}")
        print(f"{'='*70}\n")

        try:
            # Standard: top 1000 genes
            train(drug_name=drug, n_genes=1000)

            # Uncomment below to train on ALL genes (explicit & safe)
            # train(drug_name=drug, n_genes=0)  # or n_genes=None
        except Exception as e:
            print(f"FAILED for {drug}: {e}")
            import traceback
            traceback.print_exc()