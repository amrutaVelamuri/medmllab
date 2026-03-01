# MedML Lab 🧬
### Interactive Healthcare AI Learning Platform for Students

[![Live Demo](https://img.shields.io/badge/Live%20Demo-medmllab.vercel.app-00ff9d?style=for-the-badge)](https://medmllab.vercel.app)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Built with Next.js](https://img.shields.io/badge/Built%20with-Next.js-black?style=for-the-badge&logo=next.js)](https://nextjs.org)
[![AI Powered](https://img.shields.io/badge/AI-Gemini%201.5%20Flash-orange?style=for-the-badge)](https://ai.google.dev)

---

MedML Lab is a free, open-source platform that teaches students how machine learning is applied to real healthcare problems. Explore 24 medical AI domains, generate AI-powered research methodologies, and create science fair or hackathon blueprints — all in the browser, with no installation required.

**Built by a high school student, for high school students.**

---

## ✨ Features

### 🔬 Problem Explorer
Browse 24 curated medical AI problems — from pneumonia detection to Alzheimer's MRI analysis to rare disease diagnosis. Each entry includes:
- Full ML pipeline visualization
- Public dataset links (Kaggle, NIH, PhysioNet, ADNI)
- Key evaluation metrics with real benchmark numbers
- Common ML challenges specific to the domain
- Clinical stakes — *why false negatives actually kill people*

Search by keyword or filter by data type: imaging, tabular, NLP, time-series, genomics, multimodal.

### ⚗️ AI Methodology Generator
Enter any medical AI problem + your data type + skill level → get a complete research roadmap:
- Recommended model architecture with justification
- Real dataset recommendations with direct links
- Evaluation metrics with target values
- Domain-specific ML challenges
- Beginner / intermediate / advanced tips

### 📐 Research Blueprint Generator
Turn any healthcare AI idea into a full research plan for science fairs, hackathons, or papers:
- Specific, testable research question
- Measurable hypothesis
- 8-week project timeline
- Dataset recommendations
- Success metrics
- What makes the project novel

Save blueprints locally and share them via Twitter or LinkedIn.

---

## 🩺 Medical Domains Covered

| Domain | Data Type | Model |
|--------|-----------|-------|
| Pneumonia Detection | Chest X-Ray | ResNet-50 CNN |
| Diabetes Risk | EHR / Tabular | XGBoost / Random Forest |
| Alzheimer's Disease | MRI Brain Scans | 3D CNN / ViT |
| Skin Cancer | Dermoscopy Images | EfficientNet-B4 |
| Diabetic Retinopathy | Fundus Photographs | EfficientNet |
| Sepsis Prediction | ICU Time-Series | LSTM / Transformer |
| Cancer Pathology | Whole Slide Images | Vision Transformer |
| Heart Failure | EHR + ECG | Transformer / XGBoost |
| COVID-19 Detection | CT Scans + X-Rays | DenseNet |
| Stroke Detection | CT / MRI Brain | U-Net + CNN |
| Lung Cancer Screening | Low-dose CT | 3D CNN |
| Depression Detection | Text / Speech | BERT / Transformer |
| Breast Cancer | Mammography | CNN + Attention |
| Parkinson's Disease | Voice / Gait / MRI | SVM / LSTM |
| Chronic Kidney Disease | Lab Values + EHR | XGBoost |
| Drug Discovery | Molecular Graphs | Graph Neural Network |
| ICU Mortality Prediction | ICU Time-Series | LSTM / Transformer |
| Organ Segmentation | CT / MRI Volumes | U-Net / nnU-Net |
| Genomic Medicine | DNA / Gene Expression | CNN / Transformer |
| Wound Classification | Wound Photographs | MobileNet |
| Clinical NLP | Clinical Notes | BioBERT / ClinicalBERT |
| Pain Assessment | Facial Video / EEG | CNN + LSTM |
| Rare Disease Diagnosis | Multimodal | Siamese Network |
| Surgical AI | Surgical Video | CNN + Temporal Transformer |

---

## 🚀 Deploy Your Own (Free, 5 minutes)

### Prerequisites
- Node.js 18+
- A free [Google AI Studio](https://aistudio.google.com/apikey) API key (no credit card)

### Local Development
```bash
git clone https://github.com/YOUR_USERNAME/medml-lab.git
cd medml-lab
npm install
```

Create a `.env.local` file:
```
GEMINI_API_KEY=your_key_here
```

```bash
npm run dev
# Open http://localhost:3000
```

### Deploy to Vercel (Recommended)
1. Fork this repo
2. Go to [vercel.com](https://vercel.com) → New Project → Import your fork
3. Add environment variable: `GEMINI_API_KEY = your_key_here`
4. Hit Deploy → live in ~60 seconds at `yourproject.vercel.app`

The Gemini API key stays server-side — users never see it. The free tier supports **1,500 requests/day**.

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 14 |
| AI | Google Gemini 1.5 Flash |
| Styling | Pure CSS (no Tailwind) |
| Deployment | Vercel |
| Cost | **$0** |

No database. No authentication. No external services beyond Gemini.

---

## 📁 Project Structure

```
medml-lab/
├── pages/
│   ├── index.jsx          # Main app (Explorer, Methodology, Blueprint, Saved)
│   └── api/
│       └── generate.js    # Gemini API proxy — keeps key server-side
├── package.json
└── README.md
```

---

## 🎯 Who Is This For?

- **High school students** preparing science fair projects (ISEF, Regeneron, Conrad)
- **Hackathon participants** building healthcare AI projects
- **CS club members** learning real-world ML applications
- **Educators** teaching AI/ML in healthcare contexts

---

## 📄 License

MIT — free to use, modify, and deploy.

---

## 🤝 Contributing

PRs welcome. To add a new disease domain, follow the schema in `pages/index.jsx` under the `DISEASES` object and open a PR.

---

*Built by a high school student passionate about making healthcare AI research accessible to everyone.*
