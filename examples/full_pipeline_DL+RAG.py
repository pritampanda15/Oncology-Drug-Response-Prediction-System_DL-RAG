import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))

from src.data.loader import GDSCDataLoader
from src.tools.prediction_tool import DrugResponsePredictor

# Load a sample patient
loader = GDSCDataLoader("data/raw")
loader.load_expression("Cell_line_RMA_proc_basalExp.txt")
loader.load_drug_response("GDSC2_fitted_dose_response_27Oct23.xlsx")

X, y = loader.get_dataset_for_drug("Cisplatin")
X.columns = X.columns.astype(str)

# Take one sample as test patient
sample_patient = X.iloc[[0]]
true_label = y.iloc[0]

print(f"Sample patient ID: {sample_patient.index[0]}")
print(f"True label: {'Responder' if true_label == 1 else 'Non-Responder'}")

# Initialize the tool
tool = DrugResponsePredictor(
    models_dir="models",
    vector_store_dir="knowledge_base/vector_store"
)

# Get prediction with explanation
result = tool.predict_and_explain(
    drug_name="Cisplatin",
    gene_expression=sample_patient
)

print(f"\n{'='*50}")
print(f"Drug: {result['drug_name']}")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Top genes: {', '.join(result['top_genes'][:5])}")
print(f"\n{'='*50}")
print("EXPLANATION:")
print(result['explanation'])
