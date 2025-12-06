"""
Streamlit App for Drug Response Prediction with RAG Explanations

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import GDSCDataLoader
from src.tools.prediction_tool import DrugResponsePredictor
from src.data.preprocessor import prepare_splits


# Page config
st.set_page_config(
    page_title="Oncology Drug Response Prediction",
    page_icon="🧬",
    layout="wide"
)

# Title
st.title("🧬 Oncology Drug Response Prediction System")
st.markdown("### AI-Powered Drug Response Prediction with Biological Explanations")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.loader = None
    st.session_state.tool = None
    st.session_state.available_drugs = ["Cisplatin", "Docetaxel", "Paclitaxel", "Gemcitabine"]


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Load data button
    if st.button("🔄 Load Data", type="primary"):
        with st.spinner("Loading GDSC2 data..."):
            try:
                loader = GDSCDataLoader("data/raw")
                loader.load_expression("Cell_line_RMA_proc_basalExp.txt")
                loader.load_drug_response("GDSC2_fitted_dose_response_27Oct23.xlsx")

                st.session_state.loader = loader
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")

    # Initialize prediction tool
    if st.session_state.data_loaded and st.session_state.tool is None:
        with st.spinner("Initializing AI models..."):
            try:
                tool = DrugResponsePredictor(
                    models_dir="models",
                    vector_store_dir="knowledge_base/vector_store"
                )
                st.session_state.tool = tool
                st.success("✅ AI models loaded!")
            except Exception as e:
                st.error(f"❌ Error loading models: {e}")

    st.divider()

    # Info
    st.markdown("### 📊 System Info")
    if st.session_state.data_loaded:
        st.metric("Status", "Ready ✅")
        st.metric("Available Drugs", len(st.session_state.available_drugs))
    else:
        st.metric("Status", "Not Loaded ⏳")
        st.info("Click 'Load Data' to start")


# Main content
if not st.session_state.data_loaded:
    st.info("👈 Click **Load Data** in the sidebar to begin")

    # Show example
    st.markdown("### 🎯 What This Tool Does")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1️⃣ Predict Response")
        st.write("Uses deep learning to predict if a patient will respond to a drug")

    with col2:
        st.markdown("#### 2️⃣ Identify Key Genes")
        st.write("Shows which genes are most important for the prediction")

    with col3:
        st.markdown("#### 3️⃣ Explain Biology")
        st.write("Provides biological explanations using RAG and knowledge base")

else:
    # Drug selection
    selected_drug = st.selectbox(
        "🔬 Select Drug",
        st.session_state.available_drugs,
        help="Choose the chemotherapy drug to test"
    )

    # Load drug-specific data
    if selected_drug:
        with st.spinner(f"Loading {selected_drug} data..."):
            X, y = st.session_state.loader.get_dataset_for_drug(selected_drug)
            X.columns = X.columns.astype(str)

            # Store in session state
            if 'current_drug' not in st.session_state or st.session_state.current_drug != selected_drug:
                st.session_state.current_drug = selected_drug
                st.session_state.X = X
                st.session_state.y = y

    # Patient selection
    st.divider()
    st.subheader("👤 Select Patient")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Get patient list
        patient_ids = st.session_state.X.index.tolist()

        # Patient selection method
        selection_method = st.radio(
            "Selection Method",
            ["Random Patient", "Specific Patient ID"],
            horizontal=True
        )

        if selection_method == "Random Patient":
            if st.button("🎲 Pick Random Patient"):
                random_idx = np.random.randint(0, len(patient_ids))
                st.session_state.selected_patient_idx = random_idx

        else:
            selected_id = st.selectbox(
                "Patient ID (COSMIC ID)",
                patient_ids,
                help="Select a specific cell line by COSMIC ID"
            )
            st.session_state.selected_patient_idx = patient_ids.index(selected_id)

    with col2:
        if 'selected_patient_idx' in st.session_state:
            idx = st.session_state.selected_patient_idx
            patient_id = patient_ids[idx]
            true_label = st.session_state.y.iloc[idx]

            st.metric("Patient ID", patient_id)
            st.metric("True Label", "Responder ✅" if true_label == 1 else "Non-Responder ❌")

    # Prediction button
    st.divider()

    if 'selected_patient_idx' in st.session_state:
        if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
            idx = st.session_state.selected_patient_idx
            patient_id = patient_ids[idx]
            sample_patient = st.session_state.X.iloc[[idx]]

            with st.spinner("🧠 AI is analyzing gene expression patterns..."):
                try:
                    # Make prediction
                    result = st.session_state.tool.predict_and_explain(
                        drug_name=selected_drug,
                        gene_expression=sample_patient
                    )

                    # Display results
                    st.success("✅ Prediction Complete!")

                    st.divider()

                    # Prediction results
                    st.subheader("📊 Prediction Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        prediction_emoji = "✅" if result['prediction'] == "Responder" else "❌"
                        st.metric(
                            "Prediction",
                            f"{result['prediction']} {prediction_emoji}",
                            delta=f"{result['confidence']:.1%} confidence"
                        )

                    with col2:
                        true_label = st.session_state.y.iloc[idx]
                        true_label_text = "Responder" if true_label == 1 else "Non-Responder"
                        correct = (result['prediction'] == true_label_text)
                        st.metric(
                            "Actual Label",
                            f"{true_label_text} {'✅' if true_label == 1 else '❌'}",
                            delta="Correct ✓" if correct else "Incorrect ✗"
                        )

                    with col3:
                        st.metric("Patient ID", patient_id)

                    # Top genes
                    st.divider()
                    st.subheader("🧬 Top Contributing Genes")

                    genes_df = pd.DataFrame({
                        'Gene': result['top_genes'][:10],
                        'Expression': [sample_patient[g].iloc[0] for g in result['top_genes'][:10]]
                    })

                    # Create bar chart
                    st.bar_chart(genes_df.set_index('Gene'))

                    # Show table
                    with st.expander("📋 View Gene Expression Values"):
                        st.dataframe(genes_df, use_container_width=True)

                    # Explanation
                    st.divider()
                    st.subheader("📖 Biological Explanation")
                    st.info(result['explanation'])

                    # Download results
                    st.divider()

                    results_dict = {
                        "Patient ID": patient_id,
                        "Drug": selected_drug,
                        "Prediction": result['prediction'],
                        "Confidence": f"{result['confidence']:.1%}",
                        "True Label": true_label_text,
                        "Top Genes": ", ".join(result['top_genes'][:5])
                    }

                    results_df = pd.DataFrame([results_dict])

                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"prediction_{patient_id}_{selected_drug}.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🧬 Oncology Drug Response Prediction System | Powered by Deep Learning + RAG
    </div>
    """,
    unsafe_allow_html=True
)
