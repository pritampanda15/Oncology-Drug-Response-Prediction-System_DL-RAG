# Streamlit App Guide

## Quick Start

### Run the App

```bash
./run_streamlit.sh
```

Or manually:

```bash
CUDA_VISIBLE_DEVICES=1 streamlit run streamlit_app.py
```

Then open your browser to: **http://localhost:8501**

---

## Features

### 🧬 Interactive Drug Response Prediction

1. **Load Data**: Click "Load Data" in the sidebar to initialize the system
2. **Select Drug**: Choose from Cisplatin, Docetaxel, Paclitaxel, or Gemcitabine
3. **Select Patient**: Either pick a random patient or choose a specific COSMIC ID
4. **Run Prediction**: Get AI-powered predictions with biological explanations

### 📊 What You'll See

- **Prediction**: Responder or Non-Responder classification
- **Confidence Score**: Model confidence (0-100%)
- **True Label**: Actual response from GDSC2 data
- **Top Genes**: The most important genes for this prediction
- **Gene Expression Chart**: Visual representation of top contributing genes
- **Biological Explanation**: RAG-generated explanation using the knowledge base

### 📥 Download Results

Export prediction results as CSV for further analysis.

---

## Advantages Over the API

| Feature | API | Streamlit App |
|---------|-----|---------------|
| **Ease of Use** | Requires exact 1000 genes | Just select a patient |
| **Interactive** | No | Yes - real-time |
| **Visualizations** | No | Charts, metrics, tables |
| **Explanations** | JSON only | Rich formatted text |
| **Sample Data** | Manual | Built-in GDSC2 patients |

---

## System Architecture

```
User Interface (Streamlit)
    ↓
DrugResponseTool
    ↓
┌─────────────────┬─────────────────┐
│   Deep Learning │   RAG System    │
│   Predictor     │   (ChromaDB +   │
│   (PyTorch)     │   GPT-4o-mini)  │
└─────────────────┴─────────────────┘
```

---

## Troubleshooting

### App won't start
- Check if Streamlit is installed: `pip install streamlit`
- Ensure models are trained: `ls models/*.pt`
- Verify knowledge base exists: `ls knowledge_base/vector_store/`

### "No data loaded" error
- Click the "Load Data" button in the sidebar
- Check that `data/raw/` contains the required files

### RAG not working
- Ensure `.env` has `OPENAI_API_KEY` set
- Build knowledge base: `python3 scripts/build_knowledge_base.py`

---

## Tips

- **First Run**: Always click "Load Data" first
- **Best Practice**: Use "Random Patient" to test different scenarios
- **Performance**: The app caches data, so subsequent predictions are fast
- **Comparison**: Compare predictions with true labels to evaluate model performance

---

## Example Workflow

1. Start the app: `./run_streamlit.sh`
2. Click "Load Data" (wait ~10 seconds)
3. Select "Cisplatin" from drug dropdown
4. Click "Pick Random Patient"
5. Click "Run Prediction"
6. Review the prediction, genes, and biological explanation
7. Download results if needed

Enjoy! 🎉
