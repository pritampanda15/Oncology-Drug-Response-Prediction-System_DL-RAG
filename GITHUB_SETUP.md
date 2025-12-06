# GitHub Setup Guide

This guide will help you push this project to GitHub.

## 🚀 Quick Setup (5 minutes)

### Step 1: Initialize Git Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Oncology Drug Response Prediction System v1.0.0

Features:
- Deep Learning model for drug response prediction (AUC ~0.76-0.79)
- RAG-powered biological explanations
- Interactive Streamlit web interface
- Flask REST API
- Support for 4 drugs: Cisplatin, Docetaxel, Paclitaxel, Gemcitabine

Fixes:
- Gene dimension mismatch (1000 vs 218,145)
- Invalid gene name filtering (removed 318 'nan' entries)
- RAG integration with ChromaDB + GPT-4o-mini
"
```

### Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the **"+"** icon in the top-right → **"New repository"**
3. Fill in the details:
   - **Repository name**: `oncology-drug-response-prediction` (or your choice)
   - **Description**: "AI-Powered Drug Response Prediction with Biological Explanations"
   - **Visibility**: Choose Public or Private
   - **Do NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

### Step 3: Push to GitHub

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/oncology-drug-response-prediction.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 4: Update README Links

Edit `README.md` and replace:
- `https://github.com/yourusername/oncology_cds` → Your actual repo URL
- `your.email@example.com` → Your actual email

Then commit and push:

```bash
git add README.md
git commit -m "Update README with correct repository links"
git push
```

---

## 📋 Before Pushing: Checklist

Make sure you've done these:

- [ ] ✅ Deleted unnecessary files (confident_sample.py)
- [ ] ✅ Moved demo scripts to `examples/`
- [ ] ✅ Updated `.gitignore` (excludes data, models, cache)
- [ ] ✅ Created `.env.example` (template for API keys)
- [ ] ✅ Cleaned Python cache files
- [ ] ✅ Updated README.md with correct info
- [ ] ✅ Added LICENSE (MIT)
- [ ] ✅ Added CHANGELOG.md
- [ ] ✅ Removed sensitive data from `.env` (never commit this!)

---

## 🔒 Important: Sensitive Files

**NEVER commit these files** (already in `.gitignore`):

- `.env` - Contains API keys
- `data/raw/*.txt` - Large dataset files
- `data/raw/*.xlsx` - Large dataset files
- `models/*.pt` - Large model checkpoints
- `mlruns/` - MLflow tracking data

**What to commit**:

- Source code (`src/`, `scripts/`, `*.py`)
- Documentation (`*.md`)
- Configuration templates (`.env.example`, `gene_sets.yaml`)
- Small knowledge base documents
- Requirements and setup files

---

## 📦 Optional: Create a Release

After your first push:

1. Go to your GitHub repository
2. Click **"Releases"** → **"Create a new release"**
3. Tag version: `v1.0.0`
4. Release title: `v1.0.0 - Initial Release`
5. Copy-paste from `CHANGELOG.md` into the description
6. Click **"Publish release"**

---

## 🏷️ Recommended GitHub Topics

Add these topics to your repository for better discoverability:

- `drug-response-prediction`
- `deep-learning`
- `rag`
- `oncology`
- `pytorch`
- `streamlit`
- `chromadb`
- `precision-medicine`
- `cancer-research`
- `bioinformatics`
- `gdsc`
- `explainable-ai`

To add topics:
1. Go to your repository on GitHub
2. Click the **⚙️ gear icon** next to "About"
3. Add topics in the "Topics" field

---

## 📝 Optional: Add GitHub Actions (CI/CD)

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run basic imports
        run: |
          python -c "from src.data.loader import GDSCDataLoader"
          python -c "from src.models.predictor import DrugResponsePredictor"
```

---

## 🌟 Make Your README Shine

Add a screenshot or GIF of the Streamlit app:

1. Run `./run_streamlit.sh`
2. Take a screenshot of the app
3. Save as `assets/screenshot.png`
4. Add to README:

```markdown
## 🖼️ Preview

![Streamlit App](assets/screenshot.png)
```

---

## 🤝 Enable Issues and Discussions

1. Go to **Settings** → **General**
2. Under "Features":
   - ✅ Enable Issues
   - ✅ Enable Discussions (for Q&A)
3. Create issue templates for bug reports and feature requests

---

## 📊 Add Badges (Optional)

In your README, replace the placeholder badges:

```markdown
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub license](https://img.shields.io/github/license/YOUR_USERNAME/oncology-drug-response-prediction.svg)](https://github.com/YOUR_USERNAME/oncology-drug-response-prediction/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/oncology-drug-response-prediction.svg)](https://github.com/YOUR_USERNAME/oncology-drug-response-prediction/stargazers)
```

---

## ✅ You're Done!

Your repository is now:
- ✅ Clean and organized
- ✅ Well-documented
- ✅ Ready for collaboration
- ✅ GitHub-ready with proper .gitignore
- ✅ Licensed (MIT)
- ✅ Version controlled

Share your repository link and start collaborating! 🎉
