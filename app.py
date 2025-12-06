# app.py

import os
import sys
from dotenv import load_dotenv
from flask import Flask, jsonify

# --- System Path Setup ---
# Add src to the system path to allow importing modules like src.tools.prediction_tool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tools.prediction_tool import DrugResponseTool
from src.api.routes import api_blueprint 
# Import the reference to the global tool variable from the routes file
from src.api.routes import drug_response_tool as routes_tool_reference


# --- Configuration ---
load_dotenv() # Load environment variables (e.g., OPENAI_API_KEY, if used)

MODELS_DIR = "models"
VECTOR_STORE_DIR = "knowledge_base/vector_store"
# ---------------------

def create_app():
    """Factory function to create and configure the Flask application."""
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False # For cleaner, ordered API output

    # 1. Initialize the Core Tool (This handles model and KB loading)
    try:
        tool = DrugResponseTool(
            models_dir=MODELS_DIR,
            vector_store_dir=VECTOR_STORE_DIR
        )
        
        # Pass the initialized tool instance to the routes module
        # This is CRITICAL for the API endpoint to access the prediction logic
        global routes_tool_reference
        routes_tool_reference = tool
        
        # Note: We must explicitly re-assign the global variable in the imported module
        # This ensures the routes can call the tool instance.
        import src.api.routes 
        src.api.routes.drug_response_tool = tool
        
        print("✅ DrugResponseTool initialized successfully (Models & Knowledge Base loaded).")
        
    except Exception as e:
        # Log the error if initialization fails (e.g., files are missing)
        print(f"❌ ERROR: Failed to initialize DrugResponseTool. Check models/ and knowledge_base/ directories. Details: {e}")
        # The API will still run, but all prediction endpoints will return 500 errors.
        
    # 2. Register Blueprints (attaches API endpoints to the app)
    app.register_blueprint(api_blueprint)
    
    @app.route('/')
    def index():
        """Simple health check endpoint."""
        return jsonify({
            "service": "Oncology Drug Response Prediction System",
            "status": "Running",
            "api_version": "v1",
            "prediction_endpoint": "/api/predict [POST]"
        })

    return app

if __name__ == '__main__':
    # Run the application
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
