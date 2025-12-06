# src/api/routes.py

from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np

# Global reference to be populated by app.py
# This allows the API route to access the initialized tool instance
drug_response_tool = None 
api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/api/predict', methods=['POST'])
def predict_response():
    """
    API endpoint to receive gene expression data and return a drug response 
    prediction and RAG-based explanation.
    
    Expected POST body format:
    {
        "drug_name": "Cisplatin",
        "gene_expression": {
            "GENE_A": 1.2,
            "GENE_B": 3.4,
            ...
        }
    }
    """
    # 1. Check Tool Initialization
    if drug_response_tool is None:
        return jsonify({"error": "Service not initialized. DrugResponseTool is missing."}), 500

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON input data provided."}), 400

        drug_name = data.get('drug_name')
        expression_data = data.get('gene_expression')
        
        if not drug_name or not expression_data:
            return jsonify({"error": "Missing 'drug_name' or 'gene_expression' data in request."}), 400

        # 2. Convert input data to the required DataFrame format
        # The tool expects a (1, N_GENES) DataFrame with the sample ID as the index.
        try:
            sample_id = "API_SAMPLE_001" 
            
            # Create DataFrame from the dictionary, ensuring it's one row
            expression_df = pd.DataFrame(
                [expression_data], 
                index=[sample_id]
            )
            
            expression_df.columns.name = 'GENE'
            
        except Exception as e:
            return jsonify({"error": f"Invalid gene_expression format. Details: {str(e)}"}), 400

        # 3. Run the full DL + RAG pipeline
        result = drug_response_tool.predict_and_explain(
            drug_name=drug_name,
            gene_expression=expression_df
        )

        # 4. Prepare the response
        response = {
            "sample_id": sample_id,
            "drug_name": result['drug_name'],
            "prediction": result['prediction'],
            # Return confidence as a float for better API consumption
            "confidence": result['confidence'], 
            "top_genes": result['top_genes'],
            "explanation": result['explanation']
        }

        return jsonify(response), 200

    except Exception as e:
        # Catch any unexpected errors during processing
        print(f"Prediction Error: {e}")
        return jsonify({"error": "An internal server error occurred during prediction.", "details": str(e)}), 500
