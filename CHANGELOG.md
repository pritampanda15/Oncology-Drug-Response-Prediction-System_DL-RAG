# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-12-06

### 🎉 Initial Release

Complete drug response prediction system with Deep Learning and RAG-based explanations.

### ✨ Features Added

- **Deep Learning Model**: Feed-forward neural network for drug response prediction
- **RAG System**: ChromaDB + GPT-4o-mini for biological explanations
- **Streamlit App**: Interactive web interface for predictions
- **Flask API**: REST API for programmatic access
- **Multi-Drug Support**: Cisplatin, Docetaxel, Paclitaxel, Gemcitabine

### 🐛 Critical Bugs Fixed

#### 1. Gene Dimension Mismatch (Training Error)

**Issue**: Model expected 1000 genes but received 218,145 dimensions
- **Root Cause**: 318 genes had 'nan' as their name, causing dimension explosion
- **Impact**: Training failed with AssertionError
- **Files Fixed**:
  - `src/data/loader.py:50-53` - Filter invalid gene names before indexing
  - `src/data/preprocessor.py:67` - Save `self.known_genes` assignment (was missing)
  - `src/data/preprocessor.py:56-64` - Handle `n_top_genes=None` for all-gene mode

**Before**:
```
Selected 1000 genes: 0 known drug-response genes, 1000 high-variance genes
FAILED: 1000 genes != 227021 input dim
```

**After**:
```
Loaded expression: (17419, 1018)  # Removed 318 'nan' genes
Selected 1000 genes: 29 known drug-response genes, 971 high-variance genes
Final input dimension: 1000 genes ✓
```

#### 2. Missing DrugResponsePredictor Wrapper (Pipeline Error)

**Issue**: `TypeError: DrugResponsePredictor.__init__() got an unexpected keyword argument 'models_dir'`
- **Root Cause**: Name collision - neural network class vs. prediction tool wrapper
- **Impact**: Full pipeline script couldn't run
- **Files Fixed**:
  - `src/tools/prediction_tool.py:25-65` - Created `DrugResponseTool` wrapper class
  - `src/tools/prediction_tool.py:21` - Renamed neural model import to avoid collision
  - `src/tools/prediction_tool.py:421` - Added backward compatibility alias

### 🎨 Enhancements

#### Data Processing
- Removed 318 invalid gene entries (17,737 → 17,419 genes)
- Improved gene selection: prioritizes known drug-response genes
- Added robust error handling for missing genes

#### RAG Integration
- Integrated ChromaDB vector store
- Added sentence transformer embeddings (all-MiniLM-L6-v2)
- Implemented GPT-4o-mini explanation generation
- Built knowledge base with 21 document chunks

#### User Interface
- Created interactive Streamlit web app
- Added patient selection from GDSC2 dataset
- Implemented real-time visualizations (charts, tables)
- Added CSV export functionality

### 📊 Model Performance

| Drug | AUC | Accuracy | Genes Selected |
|------|-----|----------|----------------|
| Cisplatin | 0.7696 | 70.4% | 1000 (29 known + 971 variance) |
| Docetaxel | 0.7901 | 72.0% | 1000 (15 known + 985 variance) |
| Paclitaxel | 0.7850 | 71.5% | 1000 (9 known + 991 variance) |
| Gemcitabine | 0.7780 | 70.8% | 1000 (11 known + 989 variance) |

### 🏗️ Architecture Changes

**Before**:
- Inconsistent gene dimensions
- No RAG integration
- Basic prediction only

**After**:
```
Streamlit UI → DrugResponseTool → DL Model + RAG Pipeline → Prediction + Explanation
```

### 📦 Dependencies

Core packages:
- PyTorch 2.0+
- ChromaDB 0.5.0
- OpenAI SDK (GPT-4o-mini)
- Streamlit 1.52+
- scikit-learn
- pandas, numpy
- sentence-transformers

### 📝 Documentation

- Comprehensive README.md with installation and usage
- STREAMLIT_GUIDE.md for web app usage
- Code comments and docstrings
- Example scripts in `examples/`

### 🔧 Configuration

- `.gitignore` - Excludes data, models, and cache files
- `.env.example` - Template for environment variables
- `config/gene_sets.yaml` - Drug-specific gene configurations
- `requirements.txt` - Python dependencies

### 🚀 Deployment

- Streamlit app: `./run_streamlit.sh`
- Flask API: `python3 app.py`
- Training: `python3 scripts/train_model.py`
- Knowledge base: `python3 scripts/build_knowledge_base.py`

### ⚠️ Known Limitations

- Requires all 1000 genes for API predictions (use Streamlit app instead)
- Knowledge base limited to 4 drugs (expandable)
- GPU recommended for training (CPU supported but slower)

### 🙏 Acknowledgments

- GDSC database for public cancer cell line data
- Wellcome Sanger Institute
- PyTorch and ChromaDB communities

---

## Future Roadmap

### Planned Features
- [ ] Support for additional drugs
- [ ] SHAP/LIME-based feature importance
- [ ] Real patient data integration
- [ ] Confidence calibration
- [ ] A/B testing framework
- [ ] Docker containerization
- [ ] CI/CD pipeline

### Performance Improvements
- [ ] Model ensembling
- [ ] Hyperparameter optimization
- [ ] Data augmentation
- [ ] Transfer learning from larger datasets

### RAG Enhancements
- [ ] Expanded knowledge base (PubMed integration)
- [ ] Multi-document retrieval
- [ ] Citation tracking
- [ ] Feedback loop for explanation quality

---

*For detailed technical documentation, see README.md*
