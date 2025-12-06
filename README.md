# Oncology Drug Response Prediction System

<div align="center">

**AI-Powered Drug Response Prediction with Biological Explanations**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Contributing](#-contributing)

---

## 🎯 Overview

This system combines **Deep Learning** and **Retrieval-Augmented Generation (RAG)** to predict cancer drug response and provide interpretable, biologically grounded explanations. It's designed for clinical decision support, helping oncologists understand not just *what* the prediction is, but *why* the model made that prediction.

### Key Capabilities

- **Predict Drug Response**: Binary classification (Responder/Non-Responder) for chemotherapy drugs
- **Identify Key Genes**: Highlight the most important genes driving each prediction
- **Generate Biological Explanations**: Use RAG to provide context-rich explanations based on biomedical knowledge
- **Interactive Web Interface**: Streamlit app for real-time predictions and visualizations

### Supported Drugs

✅ **Cisplatin** | ✅ **Docetaxel** | ✅ **Paclitaxel** | ✅ **Gemcitabine**

---

## ✨ Features

### 🤖 Deep Learning Model

- **Architecture**: Feed-forward neural network optimized for gene expression data
- **Input**: 1,000 curated gene expression features (prioritizing drug-response genes)
- **Output**: Binary classification with confidence scores
- **Performance**: AUC ~0.76-0.79 across tested drugs

### 🧠 RAG-Powered Explanations

- **Vector Store**: ChromaDB with sentence transformers (all-MiniLM-L6-v2)
- **Knowledge Base**: Curated drug mechanism documents covering:
  - DNA damage response pathways
  - Drug resistance mechanisms
  - Key biomarkers (BRCA1/2, TP53, ERCC1, etc.)
- **LLM**: GPT-4o-mini for natural language explanation generation

### 🖥️ Interactive Streamlit App

- **Patient Selection**: Choose from 760+ cancer cell lines (GDSC2 dataset)
- **Real-Time Predictions**: Instant results with confidence scores
- **Visual Gene Analysis**: Bar charts and tables of top contributing genes
- **Biological Context**: RAG-generated explanations with cited genes
- **Export Results**: Download predictions as CSV

### 🔌 REST API

- Flask-based API for programmatic access
- JSON input/output format
- Easy integration with other systems

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                  │
│         (Patient Selection • Visualization • Export)        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   DrugResponseTool (Core)                   │
│         • Model Loading  • Prediction  • Orchestration      │
└─────────────────────────────────────────────────────────────┘
                    ↓                           ↓
    ┌───────────────────────────┐   ┌──────────────────────────┐
    │   Deep Learning Module    │   │      RAG Pipeline        │
    ├───────────────────────────┤   ├──────────────────────────┤
    │ • PyTorch Neural Network  │   │ • ChromaDB Retriever     │
    │ • 1000 gene features      │   │ • Sentence Transformers  │
    │ • Gradient-based          │   │ • GPT-4o-mini Generator  │
    │   importance scoring      │   │ • Biological context     │
    └───────────────────────────┘   └──────────────────────────┘
                    ↓                           ↓
              ┌──────────────────────────────────────┐
              │         Prediction + Explanation     │
              │  • Response class                    │
              │  • Confidence score                  │
              │  • Top genes                         │
              │  • Biological rationale              │
              └──────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, but recommended)
- ~10GB disk space (for data and models)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/oncology_cds.git
cd oncology_cds
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download GDSC2 Data

Download the following files from [GDSC](https://www.cancerrxgene.org/):

- `Cell_line_RMA_proc_basalExp.txt` (Gene expression matrix)
- `GDSC2_fitted_dose_response_27Oct23.xlsx` (Drug response data)

Place them in the `data/raw/` directory.

### 5. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

---

## ⚡ Quick Start

### Train Models

```bash
# Train all 4 drug models (takes ~10-15 minutes on GPU)
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_model.py
```

This will create model files in `models/`:
- `cisplatin_predictor.pt`
- `docetaxel_predictor.pt`
- `paclitaxel_predictor.pt`
- `gemcitabine_predictor.pt`

### Build Knowledge Base

```bash
# Create vector store from drug mechanism documents
python3 scripts/build_knowledge_base.py
```

### Launch Streamlit App

```bash
# Start the interactive web interface
./run_streamlit.sh

# Or manually:
streamlit run streamlit_app.py
```

Then open your browser to **http://localhost:8501**

---

## 📖 Usage

### Option 1: Streamlit Web App (Recommended)

1. **Launch the app**: `./run_streamlit.sh`
2. **Load data**: Click "Load Data" in the sidebar
3. **Select drug**: Choose from the dropdown (e.g., Cisplatin)
4. **Pick patient**: Use "Random Patient" or select a specific COSMIC ID
5. **Run prediction**: Click "Run Prediction"
6. **Explore results**: View prediction, confidence, top genes, and biological explanation

See [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) for detailed instructions.

### Option 2: Python Script

```python
from src.data.loader import GDSCDataLoader
from src.tools.prediction_tool import DrugResponsePredictor

# Load data
loader = GDSCDataLoader("data/raw")
loader.load_expression("Cell_line_RMA_proc_basalExp.txt")
loader.load_drug_response("GDSC2_fitted_dose_response_27Oct23.xlsx")

X, y = loader.get_dataset_for_drug("Cisplatin")
X.columns = X.columns.astype(str)

# Initialize predictor
tool = DrugResponsePredictor(
    models_dir="models",
    vector_store_dir="knowledge_base/vector_store"
)

# Make prediction
sample_patient = X.iloc[[0]]
result = tool.predict_and_explain(
    drug_name="Cisplatin",
    gene_expression=sample_patient
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Top Genes: {', '.join(result['top_genes'][:5])}")
print(f"\nExplanation:\n{result['explanation']}")
```

See `examples/full_pipeline_DL+RAG.py` for a complete example.

### Option 3: REST API

```bash
# Start the Flask API server
python3 app.py
```

Then make POST requests to `http://localhost:5000/api/predict` (requires all 1000 gene values).

---

## 📁 Project Structure

```
oncology_cds/
├── config/
│   └── gene_sets.yaml              # Drug-specific gene sets
├── data/
│   └── raw/                        # GDSC2 data files (not versioned)
├── examples/
│   └── full_pipeline_DL+RAG.py     # Demo script
├── knowledge_base/
│   ├── documents/                  # Drug mechanism documents
│   │   ├── cisplatin.txt
│   │   ├── docetaxel.txt
│   │   ├── paclitaxel.txt
│   │   └── gemcitabine.txt
│   └── vector_store/               # ChromaDB vector store
├── models/                         # Trained model checkpoints (not versioned)
├── scripts/
│   ├── train_model.py              # Model training script
│   └── build_knowledge_base.py     # Vector store builder
├── src/
│   ├── api/
│   │   └── routes.py               # Flask API routes
│   ├── data/
│   │   ├── loader.py               # GDSC2 data loader
│   │   └── preprocessor.py         # Feature selection & scaling
│   ├── models/
│   │   ├── predictor.py            # Neural network architecture
│   │   └── trainer.py              # Training utilities
│   ├── rag/
│   │   ├── ingestion.py            # Document loading & chunking
│   │   ├── retriever.py            # Vector search (ChromaDB)
│   │   └── generator.py            # LLM explanation generation
│   └── tools/
│       └── prediction_tool.py      # Main prediction + RAG integration
├── app.py                          # Flask API entry point
├── streamlit_app.py                # Streamlit web interface
├── run_streamlit.sh                # Streamlit launcher script
├── requirements.txt                # Python dependencies
├── STREAMLIT_GUIDE.md             # Streamlit usage guide
└── README.md                       # This file
```

---

## 📊 Results

### Model Performance

| Drug | AUC | Accuracy | Selected Genes |
|------|-----|----------|----------------|
| **Cisplatin** | 0.7696 | 70.4% | 1000 (29 known + 971 variance) |
| **Docetaxel** | 0.7901 | 72.0% | 1000 (15 known + 985 variance) |
| **Paclitaxel** | 0.7850 | 71.5% | 1000 (9 known + 991 variance) |
| **Gemcitabine** | 0.7780 | 70.8% | 1000 (11 known + 989 variance) |

*Performance metrics on test set (20% of data)*

### Sample Prediction Output

```
Drug: Cisplatin
Prediction: Responder
Confidence: 69.7%
Top Genes: HLA-B, HLA-DPA1, IFI30, HLA-DRA, LDHB

Biological Explanation:
The prediction of this patient as a responder to Cisplatin is supported by
the involvement of several key genes. HLA-B and HLA-DPA1 are part of the
immune response and may influence the effectiveness of Cisplatin by
modulating the immune system's ability to recognize and attack cancer cells.
GSTP1 is known for its role in drug metabolism and detoxification, where
variations can affect Cisplatin's cytotoxicity. Additionally, LDHB and VIM
are associated with cellular metabolism and epithelial-mesenchymal transition,
which can impact tumor response to chemotherapy.
```

---

## 🛠️ Technical Details

### Data Processing

- **Raw data**: 17,737 genes × 1,018 cell lines
- **After filtering**: 17,419 genes (removed 318 rows with invalid gene names)
- **Feature selection**: 1,000 genes per drug
  - Prioritizes known drug-response genes from curated gene sets
  - Fills remaining slots with high-variance genes
- **Preprocessing**: StandardScaler normalization

### Model Architecture

```python
DrugResponsePredictor(
    input_dim=1000,
    hidden_dims=[256, 128, 32],
    dropout_rate=0.5
)
```

- **Layers**: 1000 → 256 → 128 → 32 → 1
- **Activation**: ReLU
- **Dropout**: 0.5 (prevents overfitting)
- **Output**: Sigmoid (binary classification)

### Training Configuration

- **Optimizer**: Adam (lr=0.0005)
- **Weight Decay**: 0.01 (L2 regularization)
- **Scheduler**: ReduceLROnPlateau (monitors AUC)
- **Early Stopping**: Patience=15 epochs
- **Batch Size**: 64
- **Max Epochs**: 100

### RAG Pipeline

1. **Document Chunking**: 500-character chunks with overlap
2. **Embedding**: all-MiniLM-L6-v2 (384 dimensions)
3. **Retrieval**: Top 3 most relevant chunks per query
4. **Generation**: GPT-4o-mini with temperature=0.3

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`
- **Solution**: Ensure you're running scripts from the project root directory

**Issue**: `FileNotFoundError: data/raw/Cell_line_RMA_proc_basalExp.txt`
- **Solution**: Download GDSC2 data files and place them in `data/raw/`

**Issue**: RAG explanations not working
- **Solution**:
  1. Check `.env` has valid `OPENAI_API_KEY`
  2. Run `python3 scripts/build_knowledge_base.py` to build vector store

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size in `scripts/train_model.py` (line 58)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (if applicable)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **GDSC**: Genomics of Drug Sensitivity in Cancer database
- **Wellcome Sanger Institute**: For providing public cancer cell line data
- **PyTorch**: Deep learning framework
- **ChromaDB**: Vector database for RAG
- **Streamlit**: Web application framework

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact [pritam@stanford.edu](mailto:pritam@stanford.edu)

---

<div align="center">

**Built with ❤️ for advancing precision oncology**

[Report Bug](https://github.com/pritampanda15/Oncology-Drug-Response-Prediction-System_DL-RAG/issuess) • [Request Feature](https://github.com/pritampanda15/Oncology-Drug-Response-Prediction-System_DL-RAG/issues)

</div>
