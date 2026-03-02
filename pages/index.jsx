import { useState, useEffect, useMemo } from "react";
import Head from "next/head";

async function callAI(prompt) {
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data.text;
}

function extractJSON(text) {
  try {
    const match = text.match(/```json\s*([\s\S]*?)\s*```/) || text.match(/(\{[\s\S]*\})/);
    return JSON.parse(match ? match[1] : text);
  } catch { return null; }
}

const DISEASES = {
  pneumonia: {
    name: "Pneumonia", emoji: "🫁", dataType: "Chest X-Ray", model: "ResNet-50 CNN",
    tags: ["imaging", "lungs", "classification"],
    overview: "Pneumonia is a life-threatening lung infection killing ~2.5M people annually. AI screening detects it from chest X-rays with radiologist-level accuracy — vital for under-resourced hospitals.",
    whyAI: "CNNs detect consolidations, infiltrates, and opacities indicating pneumonia. One model can process thousands of X-rays overnight, enabling mass screening at scale.",
    pipeline: ["DICOM X-Ray", "Normalize + Augment", "ResNet-50", "Sigmoid Layer", "Binary Output", "Clinical Alert"],
    datasets: [
      { name: "NIH Chest X-ray14", url: "https://www.kaggle.com/datasets/nih-chest-xrays/data", size: "112,120 images", source: "NIH / Kaggle",
        meta: { samples:"112,120", features:"1024×1024px grayscale images", task:"Multi-Label Classification (14 findings)", classes:[{label:"Normal",pct:62},{label:"Pneumonia",pct:16},{label:"Infiltration",pct:12},{label:"Other findings",pct:10}] }},
      { name: "CheXpert (Stanford)", url: "https://stanfordmlgroup.github.io/competitions/chexpert/", size: "224,316 images", source: "Stanford ML Group",
        meta: { samples:"224,316", features:"Frontal + lateral view X-rays", task:"Multi-Label Classification (14 labels)", classes:[{label:"No Finding",pct:55},{label:"Pleural Effusion",pct:28},{label:"Pneumonia",pct:10},{label:"Other",pct:7}] }}
    ],
    metrics: [
      { name: "Sensitivity (Recall)", val: "~92%", desc: "% of sick patients correctly flagged" },
      { name: "Specificity", val: "~86%", desc: "% of healthy patients correctly cleared" },
      { name: "AUC-ROC", val: "~0.95", desc: "Overall discrimination ability" }
    ],
    challenges: ["Class imbalance — pneumonia is a fraction of all X-rays", "Label noise — radiologists disagree on borderline cases ~20% of the time", "Distribution shift — US-trained models may fail in low-income settings"],
    clinicalStake: "A false negative sends a pneumonia patient home without antibiotics — bacterial pneumonia can progress to sepsis and organ failure within 48 hours."
  },
  diabetes: {
    name: "Diabetes Risk", emoji: "🩸", dataType: "Patient Records (EHR)", model: "XGBoost / Random Forest",
    tags: ["tabular", "prediction", "chronic disease"],
    overview: "Over 537 million adults live with diabetes globally, and half are undiagnosed. Predictive ML can flag high-risk patients before symptoms appear.",
    whyAI: "Patient records contain dozens of correlated risk factors. Ensemble models excel at finding non-linear patterns across tabular features that simple risk scores miss.",
    pipeline: ["Patient EHR Data", "Impute + Encode", "Feature Scaling", "XGBoost / RF", "Risk Score (0-1)", "Care Team Alert"],
    datasets: [
      { name: "Pima Indians Diabetes", url: "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database", size: "768 records", source: "UCI ML Repository",
        meta: { samples:"768", features:"8 clinical features (glucose, BMI, age...)", task:"Binary Classification", classes:[{label:"No Diabetes",pct:65},{label:"Diabetes",pct:35}] }},
      { name: "CDC BRFSS Survey", url: "https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system", size: "400k+ records", source: "CDC / Kaggle",
        meta: { samples:"400,000+", features:"22 behavioral + health features", task:"Binary Classification", classes:[{label:"No Diabetes",pct:86},{label:"Diabetes",pct:14}] }}
    ],
    metrics: [
      { name: "Precision", val: "~81%", desc: "When it predicts diabetes, how often it's right" },
      { name: "Recall", val: "~78%", desc: "What % of actual diabetics are caught" },
      { name: "F1-Score", val: "~0.79", desc: "Balance of precision and recall" }
    ],
    challenges: ["Missing values — EHR data is notoriously incomplete", "Selection bias — dataset may over-represent certain demographics", "Feature leakage — some features imply diagnosis already happened"],
    clinicalStake: "A false negative means an undiagnosed patient continues developing complications — blindness, kidney failure, and limb amputations are all preventable with early intervention."
  },
  alzheimers: {
    name: "Alzheimer's", emoji: "🧠", dataType: "MRI Brain Scans", model: "3D CNN / ViT",
    tags: ["imaging", "neurology", "3D", "classification"],
    overview: "Alzheimer's affects 55M people worldwide. By the time symptoms appear, 40% of neurons may already be lost. AI detects hippocampal atrophy years before clinical diagnosis.",
    whyAI: "Subtle patterns of brain atrophy in specific regions are invisible to the naked eye but detectable by 3D CNNs trained on thousands of MRI volumes.",
    pipeline: ["3D MRI Volume", "Skull Strip + Normalize", "Voxel Resize", "3D ResNet / ViT", "4-Class Output", "Stage Report"],
    datasets: [
      { name: "ADNI (Gold Standard)", url: "https://adni.loni.usc.edu/", size: "10k+ MRI scans", source: "ADNI Consortium",
        meta: { samples:"10,000+ scans", features:"3D T1-weighted MRI volumes", task:"4-Class Classification", classes:[{label:"CN (Normal)",pct:40},{label:"MCI",pct:35},{label:"AD",pct:20},{label:"Severe AD",pct:5}] }},
      { name: "OASIS-3", url: "https://www.oasis-brains.org/", size: "2,842 sessions", source: "Washington University",
        meta: { samples:"2,842 longitudinal sessions", features:"MRI + PET + cognitive tests", task:"Binary Classification (CDR staging)", classes:[{label:"CDR 0 (Normal)",pct:72},{label:"CDR > 0 (Impaired)",pct:28}] }}
    ],
    metrics: [
      { name: "Multi-class Accuracy", val: "~85%", desc: "Across 4 stages: CN, MCI, AD, Severe" },
      { name: "AUC (vs healthy)", val: "~0.94", desc: "Discriminating early MCI from normal aging" },
      { name: "Sensitivity (MCI)", val: "~80%", desc: "Catching early Mild Cognitive Impairment" }
    ],
    challenges: ["3D data is computationally expensive — requires GPUs with 16GB+ VRAM", "Healthy aging vs early MCI is extremely subtle", "Access requires IRB approval through research portals"],
    clinicalStake: "Missing early MCI means missing the window for disease-modifying treatments. Current FDA-approved drugs only work in early stages — false negatives have irreversible consequences."
  },
  skincancer: {
    name: "Skin Cancer", emoji: "🔬", dataType: "Dermoscopy Images", model: "EfficientNet-B4",
    tags: ["imaging", "dermatology", "classification"],
    overview: "Melanoma kills ~57,000 people annually but has a 99% survival rate if caught at stage 1. AI-powered dermoscopy matches dermatologist accuracy in global screening.",
    whyAI: "EfficientNet fine-tuned on dermoscopy images achieves dermatologist-level performance, enabling expert-level screening via smartphone apps globally.",
    pipeline: ["Dermoscopy Image", "Hair Removal + Crop", "Color Normalization", "EfficientNet-B4", "7-Class Softmax", "Risk Level"],
    datasets: [
      { name: "HAM10000", url: "https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection", size: "10,015 images, 7 classes", source: "ISIC / Kaggle",
        meta: { samples:"10,015", features:"600×450px dermoscopy images", task:"7-Class Classification", classes:[{label:"Melanocytic nevi",pct:67},{label:"Melanoma",pct:11},{label:"BCC",pct:5},{label:"Other 4 types",pct:17}] }},
      { name: "ISIC 2020 Challenge", url: "https://www.kaggle.com/c/siim-isic-melanoma-classification", size: "33,126 images", source: "ISIC Archive",
        meta: { samples:"33,126", features:"High-res dermoscopy + patient metadata", task:"Binary Classification (melanoma vs benign)", classes:[{label:"Benign",pct:98},{label:"Melanoma",pct:2}] }}
    ],
    metrics: [
      { name: "Top-1 Accuracy", val: "~89%", desc: "Correct class prediction across 7 types" },
      { name: "Melanoma AUC", val: "~0.93", desc: "Detecting the most dangerous class" },
      { name: "Balanced Accuracy", val: "~82%", desc: "Accounts for extreme class imbalance" }
    ],
    challenges: ["Severe class imbalance — melanoma is <5% of all lesions", "Hair, ruler marks, and bubbles create artifacts", "Skin tone bias — training data skews toward lighter tones"],
    clinicalStake: "A missed melanoma at stage 1 (99% survival) becomes stage 4 at next visit (20% survival). Delay in diagnosis is the difference between life and death."
  },
  retinopathy: {
    name: "Diabetic Retinopathy", emoji: "👁️", dataType: "Fundus Photographs", model: "EfficientNet / InceptionV3",
    tags: ["imaging", "ophthalmology", "grading"],
    overview: "Diabetic retinopathy is the leading cause of blindness in working-age adults. AI grading deployed in India screened 11M patients, identifying 550K needing urgent care.",
    whyAI: "Fundus cameras are cheap and widely available. AI grading removes the bottleneck of specialist ophthalmologists for mass screening.",
    pipeline: ["Fundus Photo", "CLAHE Enhancement", "Green Channel Extract", "EfficientNet", "5-Grade Output", "Specialist Referral"],
    datasets: [
      { name: "EyePACS / Kaggle DR", url: "https://www.kaggle.com/c/diabetic-retinopathy-detection", size: "88,702 images", source: "EyePACS / Kaggle",
        meta: { samples:"88,702", features:"High-res fundus photographs", task:"5-Grade Ordinal Classification", classes:[{label:"Grade 0 (No DR)",pct:73},{label:"Grade 1 (Mild)",pct:7},{label:"Grade 2 (Moderate)",pct:15},{label:"Grade 3-4 (Severe+)",pct:5}] }},
      { name: "IDRiD", url: "https://idrid.grand-challenge.org/", size: "516 images + lesion masks", source: "Grand Challenge",
        meta: { samples:"516", features:"Fundus images + pixel-level lesion masks", task:"Grading + Lesion Segmentation", classes:[{label:"Grade 0-1",pct:25},{label:"Grade 2",pct:35},{label:"Grade 3",pct:25},{label:"Grade 4",pct:15}] }}
    ],
    metrics: [
      { name: "Quadratic Kappa", val: "~0.85", desc: "Competition standard — measures grade agreement" },
      { name: "Sensitivity (severe+)", val: "~90%", desc: "Catching vision-threatening grades" },
      { name: "Specificity", val: "~87%", desc: "Reducing unnecessary referrals" }
    ],
    challenges: ["Image quality varies wildly from field cameras", "Ordinal grades (0-4) require specialized loss functions", "Pseudo-labels are grader averages, not ground truth"],
    clinicalStake: "Missing severe retinopathy means missing the treatment window. Once vessels hemorrhage, vision loss is permanent."
  },
  sepsis: {
    name: "Sepsis Prediction", emoji: "🏥", dataType: "ICU Vital Signs", model: "LSTM / Transformer",
    tags: ["time-series", "ICU", "prediction", "tabular"],
    overview: "Sepsis kills ~11 million people per year. For every hour treatment is delayed, mortality rises 7%. AI early warning systems flag risk 6+ hours before clinical deterioration.",
    whyAI: "ICU patients generate thousands of data points per hour. LSTMs learn temporal patterns of deterioration invisible in static snapshots.",
    pipeline: ["ICU Vitals Stream", "Impute + Normalize", "Rolling Window (6hr)", "LSTM / Transformer", "Risk Score", "Bedside Alert"],
    datasets: [
      { name: "PhysioNet Challenge 2019", url: "https://physionet.org/content/challenge-2019/1.0.0/", size: "40,336 ICU patients", source: "PhysioNet",
        meta: { samples:"40,336 patients", features:"40 clinical variables (hourly ICU)", task:"Binary + Early Warning (6hr horizon)", classes:[{label:"No Sepsis",pct:92},{label:"Sepsis",pct:8}] }},
      { name: "MIMIC-III", url: "https://physionet.org/content/mimiciii/", size: "46k+ admissions", source: "MIT / PhysioNet",
        meta: { samples:"46,520 admissions", features:"17 time-series vitals + 100+ static vars", task:"Binary Classification (Sepsis-3 definition)", classes:[{label:"No Sepsis",pct:94},{label:"Sepsis",pct:6}] }}
    ],
    metrics: [
      { name: "AUROC", val: "~0.85", desc: "Primary competition metric" },
      { name: "Utility Score", val: "~0.42", desc: "Custom clinical utility — timing matters" },
      { name: "False Alert Rate", val: "<20%", desc: "Alert fatigue kills adoption" }
    ],
    challenges: ["Irregular time-series — labs drawn every few hours, vitals every minute", "Alert fatigue — too many false positives and nurses ignore the system", "Sepsis-2 vs Sepsis-3 definitions give completely different label distributions"],
    clinicalStake: "Every hour of delayed antibiotic treatment in sepsis increases mortality by 7%. A false negative isn't a missed diagnosis — it's a preventable death."
  },
  cancer_pathology: {
    name: "Cancer Pathology", emoji: "🧫", dataType: "Histology Slides (WSI)", model: "Vision Transformer / CNN",
    tags: ["imaging", "pathology", "oncology", "classification"],
    overview: "Digital pathology AI analyzes whole-slide images (WSIs) of biopsies to classify cancer type and grade, reducing turnaround time from days to minutes.",
    whyAI: "WSIs contain billions of pixels — no human can manually inspect every cell. AI models identify mitotic figures, necrosis, and tumor patterns across entire slides.",
    pipeline: ["Whole Slide Image", "Tile Extraction (256x256)", "Stain Normalization", "ViT / CNN", "Grade + Subtype", "Pathologist Review"],
    datasets: [
      { name: "TCGA (NCI)", url: "https://portal.gdc.cancer.gov/", size: "30k+ slides, 33 cancer types", source: "NCI GDC Portal",
        meta: { samples:"30,000+ WSIs", features:"Gigapixel H&E slides + genomics data", task:"33-Class Cancer Type Classification", classes:[{label:"Normal adjacent tissue",pct:40},{label:"Low grade tumor",pct:35},{label:"High grade tumor",pct:25}] }},
      { name: "CAMELYON17", url: "https://camelyon17.grand-challenge.org/", size: "1,000 WSIs", source: "Grand Challenge",
        meta: { samples:"1,000 lymph node WSIs", features:"Whole slide H&E stained images", task:"Binary + Patient-level Staging (pN0-pN2)", classes:[{label:"No Metastasis",pct:60},{label:"Metastasis Present",pct:40}] }}
    ],
    metrics: [
      { name: "Slide-level AUC", val: "~0.96", desc: "Distinguishing cancer vs. normal slides" },
      { name: "Grading Accuracy", val: "~88%", desc: "Correct Gleason/grade group prediction" },
      { name: "Concordance with Pathologist", val: "~91%", desc: "Agreement with expert human review" }
    ],
    challenges: ["WSIs are gigapixel images — memory management is a core engineering challenge", "Stain variation across labs causes domain shift", "Weak labels — slide-level labels, not pixel-level annotations"],
    clinicalStake: "Misgrading cancer directly affects treatment decisions — under-grading means under-treatment; over-grading leads to unnecessary chemotherapy with severe side effects."
  },
  heart_failure: {
    name: "Heart Failure", emoji: "❤️", dataType: "EHR + ECG Signals", model: "Transformer / XGBoost",
    tags: ["tabular", "time-series", "cardiology", "prediction"],
    overview: "Heart failure affects 64M people globally, with 50% dying within 5 years of diagnosis. ML models predict readmission risk, enabling targeted interventions.",
    whyAI: "Combining structured EHR data (labs, vitals, medications) with ECG signal features allows models to capture both chronic risk factors and acute deterioration signals.",
    pipeline: ["EHR + ECG Data", "Feature Engineering", "Temporal Aggregation", "Gradient Boosting", "30-day Readmission Risk", "Care Coordination"],
    datasets: [
      { name: "Heart Failure Clinical Records", url: "https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data", size: "299 patients", source: "UCI / Kaggle",
        meta: { samples:"299 patients", features:"13 clinical features (EF, creatinine...)", task:"Binary Classification (survival prediction)", classes:[{label:"Survived",pct:68},{label:"Death Event",pct:32}] }},
      { name: "MIMIC-IV", url: "https://physionet.org/content/mimiciv/", size: "500k+ hospital stays", source: "MIT / PhysioNet",
        meta: { samples:"500,000+ hospital stays", features:"200+ EHR variables + ECG waveforms", task:"30-day Readmission Prediction", classes:[{label:"No Readmission",pct:81},{label:"Readmitted <30 days",pct:19}] }}
    ],
    metrics: [
      { name: "C-statistic (AUC)", val: "~0.82", desc: "Discrimination for readmission prediction" },
      { name: "Calibration (Brier)", val: "~0.15", desc: "Predicted probabilities match actual rates" },
      { name: "NRI", val: "+12%", desc: "Net Reclassification Improvement vs clinical score" }
    ],
    challenges: ["Competing risks — patients may die before readmission, censoring data", "Administrative coding errors create noisy labels", "Model must be interpretable for clinical trust"],
    clinicalStake: "Unidentified high-risk patients are discharged without follow-up plans, leading to preventable readmissions that cost $26B annually in the US alone."
  },
  covid: {
    name: "COVID-19 Detection", emoji: "🦠", dataType: "CT Scans + X-Rays", model: "DenseNet / ResNet",
    tags: ["imaging", "infectious disease", "classification"],
    overview: "During peak COVID-19, AI models analyzed chest CT scans with >90% accuracy, triaging patients faster than PCR tests in emergency settings.",
    whyAI: "Ground-glass opacities and bilateral infiltrates in CT scans are distinctive COVID-19 signatures. AI detects these patterns faster than radiologists during surge conditions.",
    pipeline: ["CT / X-Ray", "Lung Segmentation", "ROI Extraction", "DenseNet-121", "COVID Probability", "Triage Decision"],
    datasets: [
      { name: "COVID-19 Radiography DB", url: "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database", size: "21,165 images", source: "Kaggle",
        meta: { samples:"21,165", features:"299×299px chest X-rays", task:"4-Class Classification", classes:[{label:"Normal",pct:47},{label:"COVID-19",pct:31},{label:"Viral Pneumonia",pct:14},{label:"Lung Opacity",pct:8}] }},
      { name: "SARS-CoV-2 CT Scan", url: "https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset", size: "2,482 CT scans", source: "Kaggle",
        meta: { samples:"2,482 CT slices", features:"Axial chest CT scan slices", task:"Binary Classification (COVID vs non-COVID)", classes:[{label:"Non-COVID CT",pct:50},{label:"COVID-19 CT",pct:50}] }}
    ],
    metrics: [
      { name: "Sensitivity", val: "~94%", desc: "COVID positive cases caught" },
      { name: "Specificity", val: "~91%", desc: "Non-COVID correctly cleared" },
      { name: "F1-Score", val: "~0.92", desc: "Harmonic mean of precision + recall" }
    ],
    challenges: ["COVID vs. other viral pneumonias look nearly identical on imaging", "Dataset collection bias — sicker patients are over-represented", "Model drift as variants evolve and imaging protocols change"],
    clinicalStake: "False negatives during a pandemic surge allowed infectious patients to circulate in hospitals, accelerating nosocomial transmission."
  },
  stroke: {
    name: "Stroke Detection", emoji: "⚡", dataType: "CT / MRI Brain Scans", model: "U-Net + CNN",
    tags: ["imaging", "neurology", "segmentation"],
    overview: "Stroke kills 5.5M annually. 'Time is brain' — every minute of delay destroys 1.9M neurons. AI can detect ischemic and hemorrhagic strokes in CT scans within seconds.",
    whyAI: "AI identifies penumbra regions (salvageable tissue), quantifies infarct volume, and classifies stroke type — all critical for tPA treatment decisions within the 4.5-hour window.",
    pipeline: ["CT / MRI Brain", "Skull Strip", "Atlas Registration", "U-Net Segmentation", "Infarct Volume", "tPA Decision Support"],
    datasets: [
      { name: "ISLES 2022", url: "https://isles-22.grand-challenge.org/", size: "400 MRI cases", source: "Grand Challenge",
        meta: { samples:"400 MRI cases", features:"DWI + ADC + FLAIR MRI sequences", task:"Ischemic Lesion Segmentation + Volume", classes:[{label:"Ischemic stroke",pct:70},{label:"Hemorrhagic stroke",pct:30}] }},
      { name: "PhysioNet MIMIC-III", url: "https://physionet.org/content/mimiciii/", size: "46k+ admissions", source: "PhysioNet",
        meta: { samples:"46,520 admissions", features:"Clinical variables + ICD-9 codes", task:"Binary Classification (stroke diagnosis)", classes:[{label:"No Stroke Admission",pct:95},{label:"Stroke Admission",pct:5}] }}
    ],
    metrics: [
      { name: "Dice Score (lesion)", val: "~0.78", desc: "Overlap between predicted and true lesion" },
      { name: "Volume Error", val: "<8mL", desc: "Absolute difference in infarct volume" },
      { name: "Classification AUC", val: "~0.92", desc: "Ischemic vs. hemorrhagic classification" }
    ],
    challenges: ["Acute vs. chronic lesions look similar in early CT", "Small lacunar infarcts (<10mm) are frequently missed", "Speed matters more than perfection — 95% accurate in 60 seconds beats 99% in 10 minutes"],
    clinicalStake: "A missed hemorrhagic stroke where tPA is given anyway can cause fatal brain bleeding. Misclassification is directly life-threatening."
  },
  lung_cancer: {
    name: "Lung Cancer Screening", emoji: "🫁", dataType: "Low-dose CT (LDCT)", model: "3D CNN / Nodule Detection",
    tags: ["imaging", "oncology", "detection", "3D"],
    overview: "Lung cancer has a 19% 5-year survival rate overall but 57% when caught at stage 1. LDCT screening reduces mortality by 20%, and AI automates nodule detection at scale.",
    whyAI: "3D CNNs analyze full CT volumes to detect, measure, and classify pulmonary nodules — a task requiring expert radiologists to manually scroll through hundreds of slices.",
    pipeline: ["Low-dose CT", "Lung Segmentation", "Nodule Candidate Detection", "False Positive Reduction CNN", "Malignancy Score", "Follow-up Protocol"],
    datasets: [
      { name: "LUNA16", url: "https://luna16.grand-challenge.org/", size: "888 CTs, 36k nodule annotations", source: "Grand Challenge",
        meta: { samples:"888 LDCT scans", features:"3D volumes + nodule coordinates + diameter", task:"Nodule Detection + False Positive Reduction", classes:[{label:"Benign Nodule",pct:75},{label:"Malignant Nodule",pct:25}] }},
      { name: "NLST", url: "https://cdas.cancer.gov/nlst/", size: "26,722 participants", source: "National Cancer Institute",
        meta: { samples:"26,722 participants (3 annual CTs)", features:"LDCT volumes + clinical risk factors", task:"Cancer Risk Prediction (6-year horizon)", classes:[{label:"No Cancer Detected",pct:93},{label:"Lung Cancer Diagnosed",pct:7}] }}
    ],
    metrics: [
      { name: "CPM Score (FROC)", val: "~0.90", desc: "Competition standard — sensitivity at multiple FP rates" },
      { name: "Sensitivity", val: "~94%", desc: "Nodules correctly detected" },
      { name: "False Positives/Scan", val: "<1", desc: "Critical to avoid unnecessary biopsies" }
    ],
    challenges: ["False positive nodules lead to invasive biopsies with complications", "Nodule size and density change classification thresholds", "Radiation dose must stay low, creating noisy images"],
    clinicalStake: "Missed early-stage nodules mean diagnosis at stage 4 where surgery is no longer possible — survival drops from 57% to under 5%."
  },
  mental_health: {
    name: "Depression Detection", emoji: "🧩", dataType: "Text / Speech / EHR", model: "BERT / Transformer",
    tags: ["NLP", "mental health", "classification"],
    overview: "Depression affects 280M people globally, yet 60% receive no treatment. NLP models analyzing clinical notes, social media, and speech can flag at-risk individuals.",
    whyAI: "Language patterns — word choice, syntactic complexity, first-person pronoun use — are measurable biomarkers of depression detected by transformer models.",
    pipeline: ["Clinical Notes / Speech", "Tokenization + Preprocessing", "BERT Fine-tuning", "Risk Classification", "PHQ-9 Score Estimate", "Care Navigator Alert"],
    datasets: [
      { name: "CLPsych 2015 (Twitter)", url: "https://clpsych.org/shared-task-2015/", size: "1,746 users", source: "CLPsych Workshop",
        meta: { samples:"1,746 user timelines", features:"Text posts + posting frequency patterns", task:"Binary Classification (depression screening)", classes:[{label:"Control",pct:70},{label:"Depression",pct:30}] }},
      { name: "DAIC-WOZ Interview", url: "https://dcapswoz.ict.usc.edu/", size: "189 interview sessions", source: "USC ICT",
        meta: { samples:"189 clinical interviews", features:"Audio + video + transcripts (multimodal)", task:"PHQ-8 Regression + Binary Classification", classes:[{label:"Not Depressed (PHQ<10)",pct:79},{label:"Depressed (PHQ≥10)",pct:21}] }}
    ],
    metrics: [
      { name: "F1 (depressed class)", val: "~0.73", desc: "Detecting positive cases" },
      { name: "AUC-ROC", val: "~0.82", desc: "Overall discrimination ability" },
      { name: "PHQ-9 RMSE", val: "~4.5", desc: "Score estimation accuracy" }
    ],
    challenges: ["Stigma and privacy concerns limit dataset size and diversity", "Social media data has severe selection bias", "Model must not be used for diagnosis — only triage and screening"],
    clinicalStake: "A false negative in a suicidal patient means no intervention. NLP screening tools must be validated carefully before clinical deployment."
  },
  breast_cancer: {
    name: "Breast Cancer", emoji: "🎗️", dataType: "Mammography / Histology", model: "CNN + Attention",
    tags: ["imaging", "oncology", "screening", "classification"],
    overview: "Breast cancer is the most common cancer globally. AI-assisted mammography screening catches 20% more cancers than radiologists reading alone, with half the false positives.",
    whyAI: "Deep learning models identify subtle microcalcifications and mass characteristics that predict malignancy, providing radiologists a reliable second opinion.",
    pipeline: ["Mammogram / Biopsy Slide", "Preprocessing + CLAHE", "Patch Extraction", "DenseNet / Attention CNN", "BI-RADS Category", "Radiologist Review"],
    datasets: [
      { name: "CBIS-DDSM", url: "https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset", size: "10,239 mammograms", source: "Kaggle",
        meta: { samples:"10,239 mammograms", features:"ROI patches + segmentation masks", task:"Binary Classification (malignant vs benign)", classes:[{label:"Benign",pct:60},{label:"Malignant",pct:40}] }},
      { name: "VinDr-Mammo", url: "https://vindr.ai/datasets/mammo", size: "20,000 exams", source: "VinDr",
        meta: { samples:"20,000 full exams", features:"4-view mammograms per patient", task:"BI-RADS Grading (5 assessment levels)", classes:[{label:"BI-RADS 1-2 (Normal)",pct:80},{label:"BI-RADS 3 (Probably Benign)",pct:13},{label:"BI-RADS 4-5 (Suspicious)",pct:7}] }}
    ],
    metrics: [
      { name: "AUC (malignancy)", val: "~0.94", desc: "Overall cancer detection performance" },
      { name: "Sensitivity", val: "~90%", desc: "Cancers correctly identified" },
      { name: "Recall Rate Reduction", val: "~50%", desc: "Fewer unnecessary recalls vs. single reader" }
    ],
    challenges: ["Dense breast tissue obscures lesions — model must handle 4 density categories", "Radiologist annotation inconsistency creates noisy labels", "AI must integrate with existing PACS workflow to be adopted"],
    clinicalStake: "Missed breast cancer at stage 1 has >90% survival. At stage 4, survival drops to 28%. A false negative is years of life lost."
  },
  parkinson: {
    name: "Parkinson's Disease", emoji: "🤲", dataType: "Voice / Gait / MRI", model: "SVM / LSTM / CNN",
    tags: ["multimodal", "neurology", "prediction"],
    overview: "Parkinson's affects 10M people globally. Clinical diagnosis lags biological disease onset by 10+ years. AI detects subtle voice tremors and gait anomalies years before diagnosis.",
    whyAI: "Parkinson's causes measurable changes in voice (jitter, shimmer, HNR), handwriting, and gait that are too subtle for clinical assessment but detectable by ML.",
    pipeline: ["Voice / Gait Recording", "Feature Extraction (MFCC, jitter)", "Normalization", "SVM / LSTM", "Parkinson's Probability", "Neurologist Alert"],
    datasets: [
      { name: "UCI Parkinson's Dataset", url: "https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set", size: "195 voice recordings", source: "UCI / Kaggle",
        meta: { samples:"195 voice recordings", features:"22 acoustic features (jitter, shimmer, HNR)", task:"Binary Classification", classes:[{label:"Healthy",pct:25},{label:"Parkinson's",pct:75}] }},
      { name: "mPower (Research Kit)", url: "https://www.synapse.org/#!Synapse:syn4993293", size: "10k+ participants", source: "Sage Bionetworks",
        meta: { samples:"10,000+ participants", features:"Accelerometer + voice + finger tapping", task:"Binary Classification (remote monitoring)", classes:[{label:"Control",pct:60},{label:"Parkinson's",pct:40}] }}
    ],
    metrics: [
      { name: "Classification Accuracy", val: "~93%", desc: "Parkinson's vs. healthy controls" },
      { name: "Sensitivity", val: "~96%", desc: "Parkinson's cases correctly identified" },
      { name: "Specificity", val: "~88%", desc: "Healthy individuals correctly cleared" }
    ],
    challenges: ["Voice features vary by recording device, environment, and language", "Parkinson's vs. essential tremor has significant symptom overlap", "mPower app data has self-selection bias — tech-savvy patients"],
    clinicalStake: "Earlier diagnosis allows neuroprotective treatment trials and lifestyle interventions. False negatives mean patients miss years of quality-of-life improvements."
  },
  kidney_disease: {
    name: "Chronic Kidney Disease", emoji: "🫘", dataType: "Lab Values + EHR", model: "Logistic Regression / XGBoost",
    tags: ["tabular", "prediction", "chronic disease"],
    overview: "850M people worldwide have kidney disease, with most undetected until stages 4-5. ML models predict CKD progression from routine lab panels years in advance.",
    whyAI: "CKD progression is encoded in longitudinal patterns of creatinine, eGFR, and protein in urine. ML captures non-linear interactions across lab values over time.",
    pipeline: ["Lab Values (creatinine, eGFR)", "Time-series Aggregation", "Feature Engineering", "XGBoost", "CKD Stage + Progression Risk", "Nephrology Referral"],
    datasets: [
      { name: "CKD Dataset (UCI)", url: "https://www.kaggle.com/datasets/mansoordaku/ckdisease", size: "400 patients, 24 features", source: "UCI / Kaggle",
        meta: { samples:"400 patients", features:"24 lab + clinical features", task:"Binary Classification (CKD vs non-CKD)", classes:[{label:"No CKD",pct:50},{label:"CKD",pct:50}] }},
      { name: "NHANES", url: "https://www.cdc.gov/nchs/nhanes/", size: "100k+ participants", source: "CDC NHANES",
        meta: { samples:"100,000+ participants", features:"80+ lab + lifestyle + dietary variables", task:"Binary + CKD Stage Prediction", classes:[{label:"No CKD (eGFR≥60)",pct:90},{label:"CKD (eGFR<60)",pct:10}] }}
    ],
    metrics: [
      { name: "AUC-ROC", val: "~0.97", desc: "CKD vs. non-CKD discrimination" },
      { name: "Sensitivity", val: "~99%", desc: "CKD cases identified" },
      { name: "eGFR Decline RMSE", val: "<3 mL/min/yr", desc: "Progression rate prediction accuracy" }
    ],
    challenges: ["Missing lab values — not all patients get the same tests", "CKD has 5 stages with very different clinical implications", "Confounding — diabetes and hypertension cause both CKD and abnormal labs"],
    clinicalStake: "Undetected CKD progression to end-stage renal disease requires dialysis — a life-altering, costly treatment requiring 3x/week sessions indefinitely."
  },
  drug_discovery: {
    name: "Drug Discovery", emoji: "💊", dataType: "Molecular Graphs / SMILES", model: "Graph Neural Network (GNN)",
    tags: ["GNN", "chemistry", "molecular", "regression"],
    overview: "Drug discovery takes 12+ years and costs $2.6B per approved drug. AI models predict molecular properties and binding affinities, cutting early-stage discovery from years to weeks.",
    whyAI: "GNNs represent molecules as graphs (atoms = nodes, bonds = edges) and learn property predictions directly from chemical structure, replacing expensive wet lab assays.",
    pipeline: ["SMILES String", "Molecular Graph Construction", "Atom Feature Encoding", "GNN (Message Passing)", "Property Prediction", "Virtual Screening"],
    datasets: [
      { name: "MoleculeNet", url: "http://moleculenet.org/", size: "700k+ compounds, 17 tasks", source: "Stanford / DeepChem",
        meta: { samples:"700,000+ compounds", features:"Molecular graph (atoms + bonds + charges)", task:"Multi-Task: Property Regression + Classification", classes:[{label:"Inactive compound",pct:85},{label:"Active compound",pct:15}] }},
      { name: "ChEMBL", url: "https://www.ebi.ac.uk/chembl/", size: "2.3M compounds", source: "EMBL-EBI",
        meta: { samples:"2,300,000 compounds", features:"SMILES + bioassay activity data", task:"Bioactivity Regression + Classification", classes:[{label:"Inactive (IC50>10μM)",pct:90},{label:"Active (IC50<1μM)",pct:10}] }}
    ],
    metrics: [
      { name: "AUC (bioactivity)", val: "~0.86", desc: "Active vs. inactive compound classification" },
      { name: "RMSE (binding affinity)", val: "<1.5 kcal/mol", desc: "Binding energy prediction accuracy" },
      { name: "Enrichment Factor", val: "~10x", desc: "vs. random screening baseline" }
    ],
    challenges: ["Activity cliffs — tiny structural changes cause huge property jumps", "Data imbalance — active compounds are rare in screening libraries", "Generalization to novel chemical scaffolds is poor for many models"],
    clinicalStake: "Each failed drug candidate that AI could have screened out early represents hundreds of millions in wasted research dollars and years of delay for patients awaiting treatments."
  },
  icu_mortality: {
    name: "ICU Mortality", emoji: "📈", dataType: "ICU Time-Series", model: "LSTM / Transformer",
    tags: ["time-series", "ICU", "prediction", "tabular"],
    overview: "Predicting in-hospital mortality for ICU patients enables proactive palliative care discussions and resource allocation. APACHE and SOFA scores are baseline — ML significantly outperforms them.",
    whyAI: "Longitudinal ICU data (hourly vitals, daily labs, nursing notes) contains temporal patterns of deterioration that static severity scores fundamentally miss.",
    pipeline: ["ICU Vitals + Labs", "Irregular Sampling Imputation", "Temporal Feature Extraction", "LSTM / Transformer", "48hr Mortality Risk", "ICU Team Decision Support"],
    datasets: [
      { name: "MIMIC-III Benchmarks", url: "https://github.com/YerevaNN/mimic3-benchmarks", size: "33,798 ICU stays", source: "YerevaNN / PhysioNet",
        meta: { samples:"33,798 ICU stays", features:"17 time-series clinical variables (hourly)", task:"Binary Classification (in-hospital mortality)", classes:[{label:"Survived",pct:86},{label:"Died in Hospital",pct:14}] }},
      { name: "eICU Collaborative", url: "https://eicu-crd.mit.edu/", size: "200k+ ICU stays", source: "MIT / PhysioNet",
        meta: { samples:"200,859 ICU stays", features:"200+ variables from 335 US hospitals", task:"Binary Classification + LOS Regression", classes:[{label:"Survived",pct:91},{label:"Died in Hospital",pct:9}] }}
    ],
    metrics: [
      { name: "AUC-ROC", val: "~0.87", desc: "vs. APACHE IV baseline of ~0.81" },
      { name: "AUC-PR", val: "~0.52", desc: "Precision-recall for imbalanced mortality" },
      { name: "Calibration (ECE)", val: "<0.05", desc: "Predicted probabilities are well-calibrated" }
    ],
    challenges: ["Irregular sampling — vitals every 15min, labs every 12hrs — requires careful interpolation", "Survivorship bias — sicker patients have more data points", "Ethical concerns about algorithmic end-of-life decision support"],
    clinicalStake: "Over-predicting mortality causes premature withdrawal of care. Under-predicting leads to aggressive interventions that prolong suffering. Calibration is as important as discrimination."
  },
  medical_imaging_segmentation: {
    name: "Organ Segmentation", emoji: "🩻", dataType: "CT / MRI Volumes", model: "U-Net / nnU-Net",
    tags: ["imaging", "segmentation", "3D", "radiology"],
    overview: "Organ segmentation — delineating liver, kidney, spleen in CT scans — is foundational for surgical planning and radiation therapy. Manual segmentation takes hours; AI does it in seconds.",
    whyAI: "U-Net architectures with skip connections preserve spatial resolution, enabling precise pixel-level organ boundaries essential for radiation oncology treatment planning.",
    pipeline: ["CT / MRI Volume", "Intensity Normalization", "Patch Sampling", "U-Net / nnU-Net", "Voxel-level Mask", "Surgical / RT Planning"],
    datasets: [
      { name: "Medical Segmentation Decathlon", url: "http://medicaldecathlon.com/", size: "10 tasks, 2,000+ cases", source: "Decathlon Challenge",
        meta: { samples:"2,000+ cases across 10 organ tasks", features:"CT + MRI volumes (multi-modal)", task:"Multi-Organ Semantic Segmentation", classes:[{label:"Background",pct:80},{label:"Target Organ/Tumor",pct:20}] }},
      { name: "CHAOS Challenge", url: "https://chaos.grand-challenge.org/", size: "40 CT + 120 MRI cases", source: "Grand Challenge",
        meta: { samples:"160 abdominal imaging scans", features:"CT + T1-in/T1-out/T2 MRI sequences", task:"4-Organ Segmentation (liver, kidneys, spleen)", classes:[{label:"Background",pct:78},{label:"Liver",pct:12},{label:"Spleen",pct:4},{label:"Kidneys",pct:6}] }}
    ],
    metrics: [
      { name: "Dice Score", val: "~0.95 (liver)", desc: "Overlap between predicted and true mask" },
      { name: "Hausdorff Distance", val: "<5mm", desc: "Maximum boundary deviation" },
      { name: "Surface DSC", val: "~0.93", desc: "Boundary-focused Dice metric" }
    ],
    challenges: ["Pathological organs (tumors, cysts) change shape significantly", "Adjacent organs have similar HU values — contrast is required", "nnU-Net auto-configuration is powerful but opaque to beginners"],
    clinicalStake: "Inaccurate organ segmentation in radiation therapy can result in under-dosing tumors (treatment failure) or over-dosing healthy tissue (radiation toxicity)."
  },
  genomics: {
    name: "Genomic Medicine", emoji: "🧬", dataType: "DNA / Gene Expression", model: "CNN / Transformer",
    tags: ["genomics", "multiomics", "classification", "regression"],
    overview: "Genomics AI predicts disease risk from DNA variants, identifies cancer driver mutations, and classifies tumor subtypes from gene expression, enabling precision medicine at scale.",
    whyAI: "The human genome has 3 billion base pairs — too large for manual analysis. CNNs and transformers learn sequence motifs and regulatory patterns that predict functional impact of variants.",
    pipeline: ["DNA Sequence / VCF", "One-hot Encoding / Embeddings", "Variant Filtering", "CNN / Transformer", "Pathogenicity Score", "Clinical Genetics Report"],
    datasets: [
      { name: "ClinVar", url: "https://www.ncbi.nlm.nih.gov/clinvar/", size: "1M+ variant classifications", source: "NCBI",
        meta: { samples:"1,000,000+ genetic variants", features:"DNA sequence context + functional annotations", task:"Binary Classification (pathogenicity)", classes:[{label:"Benign / Likely Benign",pct:88},{label:"Pathogenic / Likely Path.",pct:12}] }},
      { name: "TCGA Genomics", url: "https://portal.gdc.cancer.gov/", size: "33 cancer types, 20k patients", source: "NCI GDC",
        meta: { samples:"20,000+ tumor samples", features:"RNA-seq + somatic mutations + CNV", task:"33-Class Cancer Subtype Classification", classes:[{label:"BRCA (breast, most common)",pct:15},{label:"LUAD (lung adeno)",pct:10},{label:"Other 31 types",pct:75}] }}
    ],
    metrics: [
      { name: "Pathogenicity AUC", val: "~0.91", desc: "Benign vs. pathogenic variant classification" },
      { name: "AURPC", val: "~0.85", desc: "Precision-recall for rare pathogenic variants" },
      { name: "Spearman ρ (expression)", val: "~0.78", desc: "Gene expression prediction correlation" }
    ],
    challenges: ["Class imbalance — pathogenic variants are rare in population databases", "VUS (variants of uncertain significance) are ambiguous ground truth", "Linkage disequilibrium creates correlated input features"],
    clinicalStake: "Misclassifying a pathogenic variant as benign in BRCA1/2 means a patient doesn't receive prophylactic surgery — potentially missing preventable cancer."
  },
  wound_care: {
    name: "Wound Classification", emoji: "🩹", dataType: "Wound Photographs", model: "MobileNet / EfficientNet",
    tags: ["imaging", "classification", "mobile"],
    overview: "Chronic wounds affect 2% of the population and cost $50B annually. AI wound classification apps allow nurses and patients to track healing objectively via smartphone photos.",
    whyAI: "Wound size, tissue type (granulation, slough, necrosis), and moisture are objectively quantifiable from images — enabling remote monitoring without specialist visits.",
    pipeline: ["Wound Photo (smartphone)", "Background Removal", "Color Calibration", "MobileNet", "Tissue Classification + Area", "Care Plan Recommendation"],
    datasets: [
      { name: "Medetec Wound DB", url: "https://www.medetec.co.uk/files/medetec-image-databases.html", size: "1,000+ wound images", source: "Medetec",
        meta: { samples:"1,000+ images", features:"RGB wound photographs (varied conditions)", task:"Multi-Class Tissue Classification", classes:[{label:"Granulation tissue",pct:45},{label:"Slough",pct:35},{label:"Necrosis",pct:20}] }},
      { name: "AZH Wound Dataset", url: "https://github.com/uwm-bigdata/wound-classification", size: "350 annotated images", source: "UWM",
        meta: { samples:"350 annotated wound images", features:"RGB + wound area pixel measurements", task:"Multi-Class + Wound Area Regression", classes:[{label:"Granulation tissue",pct:50},{label:"Slough",pct:30},{label:"Necrosis",pct:20}] }}
    ],
    metrics: [
      { name: "Tissue Classification F1", val: "~0.82", desc: "Granulation / slough / necrosis accuracy" },
      { name: "Area Measurement MAE", val: "<8%", desc: "vs. planimetry gold standard" },
      { name: "Healing Trajectory AUC", val: "~0.79", desc: "Predicting healing vs. non-healing" }
    ],
    challenges: ["Lighting, angle, and device variation cause major domain shift", "Small dataset sizes — wound images are not widely shared", "Patient consent and privacy for wound photo sharing is complex"],
    clinicalStake: "Undetected wound deterioration (infection, necrosis) leads to amputations — 70% of non-traumatic lower limb amputations are diabetes-related and potentially preventable."
  },
  ehr_nlp: {
    name: "Clinical NLP", emoji: "📋", dataType: "Clinical Notes (Text)", model: "BioBERT / ClinicalBERT",
    tags: ["NLP", "text", "EHR", "extraction"],
    overview: "80% of clinical data is unstructured text — physician notes, discharge summaries, radiology reports. NLP extracts diagnoses, medications, and outcomes that structured EHR fields miss.",
    whyAI: "BioBERT and ClinicalBERT pre-trained on biomedical literature and clinical notes perform named entity recognition, relation extraction, and phenotyping at scale.",
    pipeline: ["Clinical Note (raw text)", "De-identification", "Tokenization (WordPiece)", "ClinicalBERT Fine-tuning", "Named Entity Tags", "Structured EHR Update"],
    datasets: [
      { name: "MIMIC-III Clinical Notes", url: "https://physionet.org/content/mimiciii/", size: "2M+ clinical notes", source: "MIT / PhysioNet",
        meta: { samples:"2,000,000+ clinical documents", features:"Free-text discharge summaries, radiology, ECG reports", task:"NER + Relation Extraction + Phenotyping", classes:[{label:"Condition Absent/Negated",pct:72},{label:"Condition Present",pct:28}] }},
      { name: "i2b2 NLP Challenges", url: "https://www.i2b2.org/NLP/DataSets/", size: "Multiple shared tasks", source: "i2b2 / Harvard",
        meta: { samples:"1,000s of annotated clinical notes", features:"De-identified inpatient clinical text", task:"NER (medications, problems, lab tests)", classes:[{label:"Non-entity tokens",pct:85},{label:"Named clinical entities",pct:15}] }}
    ],
    metrics: [
      { name: "NER F1", val: "~0.88", desc: "Named entity recognition (disorders, drugs)" },
      { name: "Relation F1", val: "~0.79", desc: "Drug-dosage relation extraction" },
      { name: "Phenotyping AUC", val: "~0.93", desc: "Condition present / absent classification" }
    ],
    challenges: ["Clinical abbreviations and acronyms are institution-specific", "Negation detection — 'no chest pain' must not extract chest pain", "De-identification is mandatory before any data sharing"],
    clinicalStake: "Missed medication allergies buried in clinical notes have caused fatal adverse drug events. NLP that misses a penicillin allergy mention is directly dangerous."
  },
  pain_assessment: {
    name: "Pain Assessment", emoji: "😣", dataType: "Facial Video / EEG", model: "CNN + LSTM",
    tags: ["video", "multimodal", "regression"],
    overview: "Pain assessment is critical in ICU patients, neonates, and those with cognitive impairment who cannot self-report. AI analyzes facial action units and physiological signals to quantify pain objectively.",
    whyAI: "Facial action coding system (FACS) features extracted by CNNs from video frames, combined with LSTM temporal modeling, predict pain intensity scores non-invasively.",
    pipeline: ["Face Video / EEG", "Face Detection + Alignment", "Action Unit Extraction", "CNN + LSTM", "Pain Intensity Score (0-10)", "Medication Alert"],
    datasets: [
      { name: "UNBC-McMaster Shoulder Pain", url: "https://www.pitt.edu/~jeffcohn/UNBC-McMaster.htm", size: "200 video sequences", source: "Pitt / UBC",
        meta: { samples:"200 video sequences", features:"Facial video + FACS action unit labels", task:"Pain Intensity Regression (PSPI 0-16 scale)", classes:[{label:"No/Minimal Pain (0-2)",pct:40},{label:"Mild Pain (3-6)",pct:35},{label:"Severe Pain (7+)",pct:25}] }},
      { name: "BioVid Heat Pain DB", url: "https://www.nit.ovgu.de/nit_media/Forschung/Datenbanken/BioVid_Heat_Pain_Database-p-12.html", size: "87 subjects", source: "Univ. Magdeburg",
        meta: { samples:"87 subjects (controlled heat stimuli)", features:"Video + EEG + ECG + skin conductance", task:"Pain Level Regression (4 stimulation levels)", classes:[{label:"Baseline (no pain)",pct:50},{label:"Low-intensity stimulation",pct:30},{label:"High-intensity stimulation",pct:20}] }}
    ],
    metrics: [
      { name: "ICC (intensity)", val: "~0.71", desc: "Inter-rater correlation for pain scoring" },
      { name: "Pain Detection AUC", val: "~0.85", desc: "Pain vs. no-pain classification" },
      { name: "Sequence PCC", val: "~0.67", desc: "Pearson correlation with PSPI scores" }
    ],
    challenges: ["Small datasets — pain research has significant ethical barriers to data collection", "Racial and gender bias in facial action coding systems", "Privacy concerns with continuous video monitoring in clinical settings"],
    clinicalStake: "Untreated pain in non-verbal ICU patients causes physiological stress, delayed healing, and long-term psychological trauma. Over-treatment causes respiratory depression."
  },
  rare_disease: {
    name: "Rare Disease Diagnosis", emoji: "🔍", dataType: "Multimodal (images, labs, text)", model: "Siamese Network / Zero-shot",
    tags: ["multimodal", "classification", "rare", "few-shot"],
    overview: "There are 7,000+ rare diseases, and average diagnosis takes 4-7 years across 7 specialists. AI trained on phenotype-genotype relationships can narrow differentials dramatically.",
    whyAI: "Siamese networks and zero-shot learning handle the extreme class imbalance of rare diseases. Facial analysis AI (Face2Gene) classifies hundreds of syndromes from clinical photos.",
    pipeline: ["Patient Photos + Labs + HPO Terms", "Phenotype Embedding", "Siamese / Zero-shot Network", "Syndrome Similarity Ranking", "Top-K Differential", "Geneticist Review"],
    datasets: [
      { name: "Orphanet", url: "https://www.orpha.net/", size: "6,000+ rare diseases", source: "INSERM / EU",
        meta: { samples:"6,000+ disease entries", features:"Clinical descriptions + gene-disease links", task:"Multi-Class Knowledge Graph Classification", classes:[{label:"Ultra-rare (<1 per million)",pct:65},{label:"Very rare (1-9 per million)",pct:20},{label:"Rare (1-5 per 10,000)",pct:15}] }},
      { name: "GestaltMatcher DB", url: "https://gestaltmatcher.org/", size: "10k+ patient images", source: "Univ. Bonn",
        meta: { samples:"10,000+ patient facial images", features:"Facial photographs + HPO phenotype terms", task:"Multi-Class (1,000+ distinct syndromes)", classes:[{label:"Most common syndrome (5%)",pct:5},{label:"2nd most common (4%)",pct:4},{label:"All other syndromes",pct:91}] }}
    ],
    metrics: [
      { name: "Top-1 Accuracy", val: "~35%", desc: "Correct syndrome as first prediction" },
      { name: "Top-10 Accuracy", val: "~83%", desc: "Correct syndrome in top 10 differential" },
      { name: "Diagnostic Yield", val: "+40%", desc: "Increase vs. clinical suspicion alone" }
    ],
    challenges: ["Extreme class imbalance — some diseases have <10 known cases worldwide", "Ground truth requires genetic confirmation, creating long labeling delays", "Privacy — rare disease patients are often identifiable from phenotype data alone"],
    clinicalStake: "4-7 year diagnostic odyssey for rare disease patients causes preventable organ damage, inappropriate treatments, and devastating psychological burden on families."
  },
  surgical_ai: {
    name: "Surgical AI", emoji: "🔧", dataType: "Surgical Video", model: "CNN + Temporal Transformer",
    tags: ["video", "segmentation", "robotics"],
    overview: "AI in surgical robotics tracks tool position, recognizes operative phases, and flags skill deviations in real-time — reducing complications and enabling remote surgical training.",
    whyAI: "CNNs detect tools and anatomy in video frames; temporal models recognize the sequence of surgical phases. This enables automated skill assessment and real-time guidance.",
    pipeline: ["Endoscopic Video", "Frame Extraction", "Tool + Tissue Detection", "Temporal CNN / Transformer", "Phase Recognition + Skill Score", "Surgeon Feedback"],
    datasets: [
      { name: "Cholec80", url: "https://camma.unistra.fr/datasets/", size: "80 cholecystectomy videos", source: "IHU Strasbourg",
        meta: { samples:"80 full surgical videos (17+ hours)", features:"25fps endoscopic frames + tool presence labels", task:"7-Phase Recognition + Tool Detection", classes:[{label:"Phases 1-3 (prep/calot)",pct:40},{label:"Phases 4-5 (dissection)",pct:35},{label:"Phases 6-7 (extraction)",pct:25}] }},
      { name: "CholecT50", url: "https://github.com/CAMMA-public/cholect50", size: "50 videos + triplet labels", source: "IHU Strasbourg",
        meta: { samples:"50 videos with dense per-frame annotation", features:"Instrument-verb-target action triplets", task:"Surgical Action Triplet Recognition", classes:[{label:"No active action triplet",pct:65},{label:"Active surgical action",pct:35}] }}
    ],
    metrics: [
      { name: "Phase Recognition Acc", val: "~92%", desc: "Correct surgical phase classification" },
      { name: "Tool Detection mAP", val: "~75%", desc: "Instrument detection accuracy" },
      { name: "Skill Assessment ICC", val: "~0.73", desc: "Agreement with expert human ratings" }
    ],
    challenges: ["Smoke, blood, and occlusion severely degrade video quality during critical moments", "Surgical variation between surgeons — same phase looks very different", "High stakes deployment — model errors during live surgery are unacceptable"],
    clinicalStake: "Undetected intraoperative errors (bile duct injury, vascular damage) that AI could have flagged lead to life-threatening complications and repeat surgeries."
  }
};

const ALL_TAGS = ["all","imaging","tabular","NLP","time-series","genomics","3D","multimodal","oncology","neurology","cardiology","ICU"];

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
:root{--bg:#020c17;--s1:#061526;--s2:#0a1f35;--s3:#0f2944;--border:rgba(0,200,150,0.15);--border2:rgba(0,200,150,0.3);--green:#00ff9d;--blue:#00b8ff;--red:#ff4d6d;--yellow:#ffd166;--text:#d4efff;--muted:#567a9a;--dim:#2a4a6a;}
body{background:var(--bg);color:var(--text);font-family:'Outfit',sans-serif;min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;background:radial-gradient(ellipse 80% 50% at 50% -10%,rgba(0,184,255,0.07) 0%,transparent 60%),radial-gradient(ellipse 60% 40% at 80% 80%,rgba(0,255,157,0.04) 0%,transparent 50%);}
.app{position:relative;z-index:1;max-width:1080px;margin:0 auto;padding:0 20px 100px;}
.nav{display:flex;align-items:center;justify-content:space-between;padding:20px 0;border-bottom:1px solid var(--border);margin-bottom:32px;}
.logo{font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:800;letter-spacing:1px;}
.logo .dot{color:var(--green);}
.logo .sub{font-size:0.7rem;font-family:'JetBrains Mono',monospace;color:var(--muted);display:block;letter-spacing:3px;margin-top:-4px;text-transform:uppercase;}
.tabs{display:flex;gap:2px;background:var(--s1);border:1px solid var(--border);border-radius:10px;padding:3px;}
.tab{font-family:'JetBrains Mono',monospace;font-size:0.68rem;padding:7px 16px;border-radius:7px;cursor:pointer;border:none;background:transparent;color:var(--muted);transition:all .2s;letter-spacing:.5px;text-transform:uppercase;white-space:nowrap;}
.tab.active{background:var(--green);color:var(--bg);font-weight:500;}
.tab:not(.active):hover{color:var(--green);}
.hero-eyebrow{font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--green);letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:8px;}
.hero-eyebrow::before{content:'';width:6px;height:6px;background:var(--green);border-radius:50%;animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.card{background:var(--s1);border:1px solid var(--border);border-radius:14px;padding:24px;margin-bottom:16px;}
.card-head{font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;font-weight:700;letter-spacing:.5px;margin-bottom:4px;display:flex;align-items:center;gap:10px;}
.card-sub{color:var(--muted);font-size:0.82rem;margin-bottom:20px;}
.search-bar{display:flex;align-items:center;gap:10px;background:var(--s2);border:1px solid var(--border2);border-radius:10px;padding:10px 16px;margin-bottom:14px;}
.search-bar input{background:transparent;border:none;outline:none;color:var(--text);font-family:'Outfit',sans-serif;font-size:0.92rem;flex:1;}
.search-bar input::placeholder{color:var(--muted);}
.search-count{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--muted);white-space:nowrap;}
.tag-row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px;}
.filter-tag{font-family:'JetBrains Mono',monospace;font-size:0.65rem;padding:5px 10px;border-radius:6px;cursor:pointer;border:1px solid var(--border);background:var(--s2);color:var(--muted);transition:all .2s;text-transform:uppercase;letter-spacing:.5px;}
.filter-tag:hover,.filter-tag.active{border-color:var(--green);color:var(--green);background:rgba(0,255,157,0.06);}
.disease-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px;max-height:520px;overflow-y:auto;padding-right:4px;}
.disease-grid::-webkit-scrollbar{width:4px;}
.disease-grid::-webkit-scrollbar-track{background:var(--s2);border-radius:2px;}
.disease-grid::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
.disease-btn{background:var(--s2);border:1px solid var(--border);border-radius:10px;padding:16px;cursor:pointer;text-align:left;transition:all .2s;}
.disease-btn:hover{transform:translateY(-2px);border-color:var(--border2);}
.disease-btn.selected{border-color:var(--green);background:rgba(0,255,157,0.05);}
.disease-emoji{font-size:1.5rem;margin-bottom:8px;display:block;}
.disease-name{font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;margin-bottom:4px;line-height:1.1;}
.disease-type{font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--green);text-transform:uppercase;letter-spacing:.5px;}
.disease-model{font-size:0.72rem;color:var(--muted);margin-top:4px;}
.no-results{grid-column:1/-1;text-align:center;padding:40px;color:var(--muted);font-size:.9rem;}
.pipeline{display:flex;align-items:center;overflow-x:auto;padding:14px 0;margin:16px 0;scrollbar-width:none;}
.pipeline::-webkit-scrollbar{display:none;}
.pipe-node{background:var(--s2);border:1px solid var(--border2);border-radius:7px;padding:8px 12px;font-family:'JetBrains Mono',monospace;font-size:0.65rem;white-space:nowrap;animation:slideIn .4s both;flex-shrink:0;}
@keyframes slideIn{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:translateX(0)}}
.pipe-arrow{color:var(--dim);font-size:.9rem;margin:0 5px;flex-shrink:0;}
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px;}
.info-panel{background:var(--s2);border:1px solid var(--border);border-radius:10px;padding:14px;}
.info-panel.full{grid-column:1/-1;}
.info-panel-label{font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--green);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}
.info-panel p{font-size:0.83rem;line-height:1.65;color:var(--text);font-weight:300;}
.metric-row{display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--border);}
.metric-row:last-child{border:none;}
.metric-name{font-family:'JetBrains Mono',monospace;font-size:0.7rem;min-width:160px;}
.metric-val{font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:var(--green);font-weight:500;min-width:60px;}
.metric-desc{font-size:0.78rem;color:var(--muted);}
.dataset-card{display:flex;align-items:center;justify-content:space-between;background:var(--s3);border:1px solid var(--border);border-radius:8px;padding:10px 14px;margin-bottom:8px;cursor:pointer;transition:all .2s;}
.dataset-card:hover{border-color:var(--blue);}
.dataset-name{font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--blue);}
.dataset-meta{font-size:0.7rem;color:var(--muted);margin-top:2px;}
.challenge-item{display:flex;gap:10px;padding:7px 0;border-bottom:1px solid var(--border);font-size:0.82rem;line-height:1.5;}
.challenge-item:last-child{border:none;}
.challenge-num{font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:var(--red);background:rgba(255,77,109,0.1);border-radius:4px;padding:2px 5px;height:fit-content;flex-shrink:0;}
.stake-box{background:rgba(255,77,109,0.06);border:1px solid rgba(255,77,109,0.3);border-radius:10px;padding:12px 14px;margin-top:12px;display:flex;gap:10px;}
.stake-text{font-size:0.82rem;line-height:1.6;color:#ffb3c0;}
.field{display:flex;flex-direction:column;gap:6px;margin-bottom:14px;}
.field label{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;}
.field input,.field select,.field textarea{background:var(--s2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:'Outfit',sans-serif;font-size:0.9rem;padding:10px 14px;outline:none;transition:border .2s;width:100%;}
.field input:focus,.field select:focus,.field textarea:focus{border-color:var(--green);}
.field select option{background:var(--s2);}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.btn{display:inline-flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;font-size:0.75rem;letter-spacing:.5px;text-transform:uppercase;padding:11px 22px;border-radius:9px;border:none;cursor:pointer;transition:all .2s;font-weight:500;}
.btn-green{background:var(--green);color:var(--bg);}
.btn-green:hover{background:#33ffb2;transform:translateY(-1px);box-shadow:0 4px 20px rgba(0,255,157,.3);}
.btn-green:disabled{opacity:.35;cursor:not-allowed;transform:none;box-shadow:none;}
.btn-ghost{background:transparent;border:1px solid var(--border2);color:var(--green);}
.btn-ghost:hover{background:rgba(0,255,157,.08);}
.spinner{width:16px;height:16px;border:2px solid rgba(0,255,157,.2);border-top-color:var(--green);border-radius:50%;animation:spin .7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.out-section{margin-top:20px;animation:fadeUp .4s both;}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.out-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.out-card{background:var(--s2);border:1px solid var(--border);border-radius:10px;padding:16px;animation:fadeUp .4s both;}
.out-card.accent{border-color:rgba(0,255,157,.3);background:rgba(0,255,157,.04);}
.out-label{font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:var(--green);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;}
.out-title{font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:6px;}
.out-body{font-size:0.82rem;color:var(--muted);line-height:1.65;}
.out-link{display:inline-flex;align-items:center;gap:6px;font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--blue);text-decoration:none;margin-top:8px;border:1px solid rgba(0,184,255,.3);border-radius:6px;padding:5px 10px;transition:all .2s;}
.out-link:hover{background:rgba(0,184,255,.1);}
.shimmer{background:linear-gradient(90deg,var(--s2) 25%,var(--s3) 50%,var(--s2) 75%);background-size:200% 100%;animation:shimmer 1.5s infinite;border-radius:8px;}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
.shimmer-line{height:14px;margin-bottom:8px;}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;}
.chip{font-size:0.78rem;padding:6px 12px;border-radius:7px;cursor:pointer;border:1px solid var(--border);background:var(--s2);color:var(--muted);transition:all .2s;}
.chip:hover,.chip.active{border-color:var(--green);color:var(--green);background:rgba(0,255,157,.06);}
.timeline .tl-item{display:flex;gap:14px;padding:10px 0;position:relative;}
.timeline .tl-item:not(:last-child)::after{content:'';position:absolute;left:15px;top:36px;bottom:0;width:1px;background:var(--border2);}
.tl-dot{width:30px;height:30px;background:var(--s3);border:2px solid var(--green);border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--green);flex-shrink:0;z-index:1;}
.tl-week{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--muted);}
.tl-task{font-size:0.85rem;margin-top:2px;}
.saved-card{background:var(--s2);border:1px solid var(--border);border-radius:10px;padding:14px 18px;display:flex;align-items:center;justify-content:space-between;cursor:pointer;transition:all .2s;margin-bottom:10px;}
.saved-card:hover{border-color:var(--border2);}
.ds-panel{background:var(--s2);border:1px solid rgba(0,184,255,.25);border-radius:12px;padding:18px;margin-top:0;margin-bottom:8px;animation:fadeUp .3s both;}
.ds-panel-title{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:var(--blue);text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;}
.ds-meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px;}
.ds-meta-item{background:var(--s3);border-radius:8px;padding:10px 12px;}
.ds-meta-label{font-family:'JetBrains Mono',monospace;font-size:.58rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px;}
.ds-meta-val{font-size:.85rem;font-weight:500;color:var(--text);}
.dist-label{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--muted);margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px;}
.dist-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.dist-name{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:var(--text);min-width:140px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.dist-bar-wrap{flex:1;background:var(--s3);border-radius:4px;height:8px;overflow:hidden;}
.dist-bar{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--blue),var(--green));transition:width .6s ease;}
.dist-pct{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:var(--green);min-width:32px;text-align:right;}
@media(max-width:640px){.form-row{grid-template-columns:1fr}.info-grid{grid-template-columns:1fr}.out-grid{grid-template-columns:1fr}.tabs{display:none}.disease-grid{grid-template-columns:repeat(2,1fr)}}
`;

export default function App() {
  const [tab, setTab] = useState(0);
  const TABS = ["🔬 Explorer","⚗️ Methodology","📐 Blueprint","💾 Saved"];
  return (
    <>
      <Head><title>MedML Lab — Healthcare AI for Students</title></Head>
      <style>{CSS}</style>
      <div className="app">
        <nav className="nav">
          <div className="logo">MedML<span className="dot">.</span>Lab<span className="sub">Healthcare AI for Students</span></div>
          <div className="tabs">{TABS.map((t,i)=><button key={t} className={`tab ${tab===i?"active":""}`} onClick={()=>setTab(i)}>{t}</button>)}</div>
        </nav>
        <div style={{marginBottom:28}}>
          <div className="hero-eyebrow">Free · Open Source · No Login</div>
          <h1 style={{fontFamily:"'Barlow Condensed',sans-serif",fontSize:"clamp(2rem,5vw,3.2rem)",fontWeight:800,lineHeight:.95,letterSpacing:-1,marginBottom:12}}>
            From idea to <span style={{color:"var(--green)"}}>research-ready</span><br/>in minutes.
          </h1>
          <div className="tabs" style={{display:"inline-flex"}}>{TABS.map((t,i)=><button key={t} className={`tab ${tab===i?"active":""}`} onClick={()=>setTab(i)}>{t}</button>)}</div>
        </div>
        {tab===0 && <Explorer />}
        {tab===1 && <Methodology />}
        {tab===2 && <Blueprint />}
        {tab===3 && <Saved />}
      </div>
    </>
  );
}

function DatasetPanel({ ds }) {
  const meta = ds?.meta;
  if (!meta) return null;
  return (
    <div className="ds-panel">
      <div className="ds-panel-title">📊 {ds.name}</div>
      <div className="ds-meta-grid">
        <div className="ds-meta-item"><div className="ds-meta-label">Samples</div><div className="ds-meta-val">{meta.samples}</div></div>
        <div className="ds-meta-item"><div className="ds-meta-label">Features</div><div className="ds-meta-val">{meta.features}</div></div>
        <div className="ds-meta-item" style={{gridColumn:"1/-1"}}><div className="ds-meta-label">ML Task</div><div className="ds-meta-val">{meta.task}</div></div>
      </div>
      <div className="dist-label">Class Distribution</div>
      {meta.classes.map((c,i)=>(
        <div key={i} className="dist-row">
          <div className="dist-name">{c.label}</div>
          <div className="dist-bar-wrap"><div className="dist-bar" style={{width:`${c.pct}%`}}/></div>
          <div className="dist-pct">{c.pct}%</div>
        </div>
      ))}
    </div>
  );
}

function Explorer() {
  const [search, setSearch] = useState("");
  const [activeTag, setActiveTag] = useState("all");
  const [sel, setSel] = useState(null);
  const [selDs, setSelDs] = useState(null);

  const filtered = useMemo(()=>{
    return Object.entries(DISEASES).filter(([key,d])=>{
      const matchesTag = activeTag==="all" || d.tags.includes(activeTag);
      const q = search.toLowerCase();
      const matchesSearch = !q || d.name.toLowerCase().includes(q) || d.dataType.toLowerCase().includes(q) || d.tags.some(t=>t.includes(q)) || d.overview.toLowerCase().includes(q);
      return matchesTag && matchesSearch;
    });
  },[search,activeTag]);

  const d = sel ? DISEASES[sel] : null;

  return (
    <div>
      <div className="card">
        <div className="card-head">🔬 Problem Explorer</div>
        <div className="card-sub">Browse 24 medical AI problems. Click any disease to see pipeline, datasets, metrics, and clinical stakes.</div>
        <div className="search-bar">
          <span>🔍</span>
          <input placeholder="Search diseases, data types, methods..." value={search} onChange={e=>{setSearch(e.target.value);setSel(null);setSelDs(null);}}/>
          <span className="search-count">{filtered.length} / {Object.keys(DISEASES).length}</span>
        </div>
        <div className="tag-row">
          {ALL_TAGS.map(tag=>(
            <div key={tag} className={`filter-tag ${activeTag===tag?"active":""}`} onClick={()=>{setActiveTag(tag);setSel(null);setSelDs(null);}}>{tag}</div>
          ))}
        </div>
        <div className="disease-grid">
          {filtered.length===0 ? <div className="no-results">No diseases match your search.</div> :
            filtered.map(([key,dis])=>(
              <button key={key} className={`disease-btn ${sel===key?"selected":""}`} onClick={()=>{setSel(sel===key?null:key);setSelDs(null);}}>
                <span className="disease-emoji">{dis.emoji}</span>
                <div className="disease-name">{dis.name}</div>
                <div className="disease-type">{dis.dataType}</div>
                <div className="disease-model">{dis.model}</div>
              </button>
            ))}
        </div>
      </div>

      {d && (
        <div style={{animation:"fadeUp .3s both"}}>
          <div className="card" style={{borderColor:"rgba(0,255,157,.2)"}}>
            <div className="card-head">{d.emoji} {d.name}</div>
            <div className="pipeline">
              {d.pipeline.map((step,i)=>(
                <span key={i}>
                  <span className="pipe-node" style={{animationDelay:`${i*.07}s`}}>{step}</span>
                  {i<d.pipeline.length-1 && <span className="pipe-arrow">→</span>}
                </span>
              ))}
            </div>
            <div className="info-grid">
              <div className="info-panel"><div className="info-panel-label">Overview</div><p>{d.overview}</p></div>
              <div className="info-panel"><div className="info-panel-label">Why AI Helps</div><p>{d.whyAI}</p></div>
            </div>
          </div>

          <div className="card">
            <div className="card-head">📊 Evaluation Metrics</div>
            {d.metrics.map(m=>(
              <div key={m.name} className="metric-row">
                <div className="metric-name">{m.name}</div>
                <div className="metric-val">{m.val}</div>
                <div className="metric-desc">{m.desc}</div>
              </div>
            ))}
          </div>

          <div className="card">
            <div className="card-head">📦 Public Datasets</div>
            <div style={{fontSize:".75rem",color:"var(--muted)",marginBottom:12,fontFamily:"'JetBrains Mono',monospace"}}>↓ Click a dataset to see its unique stats + class distribution</div>
            {d.datasets.map(ds=>(
              <div key={ds.name}>
                <div
                  className="dataset-card"
                  style={{borderColor:selDs===ds.name?"var(--blue)":""}}
                  onClick={()=>setSelDs(selDs===ds.name?null:ds.name)}
                >
                  <div>
                    <div className="dataset-name">{ds.name}</div>
                    <div className="dataset-meta">{ds.size} · {ds.source}</div>
                  </div>
                  <div style={{display:"flex",gap:10,alignItems:"center"}}>
                    <a href={ds.url} target="_blank" rel="noreferrer" onClick={e=>e.stopPropagation()} style={{color:"var(--blue)",fontSize:".9rem",textDecoration:"none"}}>↗</a>
                    <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:".65rem",color:"var(--muted)"}}>{selDs===ds.name?"▲":"▼"}</span>
                  </div>
                </div>
                {selDs===ds.name && <DatasetPanel ds={ds} />}
              </div>
            ))}
          </div>

          <div className="card">
            <div className="card-head">⚠️ ML Challenges</div>
            {d.challenges.map((c,i)=>(
              <div key={i} className="challenge-item"><div className="challenge-num">0{i+1}</div><div>{c}</div></div>
            ))}
            <div className="stake-box">
              <div style={{fontSize:"1.1rem"}}>🚨</div>
              <div className="stake-text"><strong style={{color:"var(--red)"}}>Clinical Stakes: </strong>{d.clinicalStake}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Methodology() {
  const [disease, setDisease] = useState("");
  const [dataType, setDataType] = useState("medical-images");
  const [level, setLevel] = useState("beginner");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const EXAMPLES = ["Detect pneumonia from chest X-rays","Predict sepsis in ICU patients","Classify skin lesions","Predict 30-day hospital readmission","Detect depression from clinical notes"];

  async function generate() {
    if (!disease.trim()) return;
    setLoading(true); setResult(null); setErr("");
    try {
      const text = await callAI(`You are a healthcare ML research mentor. Generate a methodology for: "${disease}" using ${dataType}, skill level: ${level}.
Return ONLY valid JSON (no markdown fences):
{"model":{"name":"string","architecture":"string","reason":"string"},"dataset":{"name":"string","url":"string","size":"string","note":"string"},"dataset2":{"name":"string","url":"string","size":"string"},"metrics":[{"name":"string","target":"string","why":"string"},{"name":"string","target":"string","why":"string"},{"name":"string","target":"string","why":"string"}],"challenges":["string","string","string"],"clinicalStake":"string","tips":["string","string"]}`);
      const parsed = extractJSON(text);
      if (!parsed) throw new Error("Parse error — please try again.");
      setResult(parsed);
    } catch(e) { setErr(e.message); }
    setLoading(false);
  }

  return (
    <div>
      <div className="card">
        <div className="card-head">⚗️ AI Methodology Generator</div>
        <div className="card-sub">Describe any medical AI problem — get a full research roadmap with model, datasets, and metrics.</div>
        <div className="chips">{EXAMPLES.map(e=><div key={e} className={`chip ${disease===e?"active":""}`} onClick={()=>setDisease(e)}>{e}</div>)}</div>
        <div className="field"><label>Disease / Problem</label><input placeholder="Describe your medical AI problem..." value={disease} onChange={e=>setDisease(e.target.value)}/></div>
        <div className="form-row">
          <div className="field"><label>Data Type</label>
            <select value={dataType} onChange={e=>setDataType(e.target.value)}>
              <option value="medical-images">Medical Images (X-ray, MRI, CT)</option>
              <option value="patient-health-records">Patient Health Records (EHR)</option>
              <option value="genomics">Genomics / Gene Expression</option>
              <option value="wearable-sensor-data">Wearable / Sensor Data</option>
              <option value="clinical-notes-nlp">Clinical Notes (NLP)</option>
              <option value="icu-time-series">ICU Time-Series Vitals</option>
              <option value="surgical-video">Surgical Video</option>
            </select>
          </div>
          <div className="field"><label>Skill Level</label>
            <select value={level} onChange={e=>setLevel(e.target.value)}>
              <option value="beginner">Beginner (Python basics)</option>
              <option value="intermediate">Intermediate (knows sklearn)</option>
              <option value="advanced">Advanced (deep learning)</option>
            </select>
          </div>
        </div>
        {err && <div style={{color:"var(--red)",fontSize:".8rem",marginBottom:12}}>{err}</div>}
        <button className="btn btn-green" onClick={generate} disabled={loading||!disease.trim()}>
          {loading?<><div className="spinner"/>Generating...</>:"Generate Methodology →"}
        </button>
      </div>
      {loading && <div className="card">{[100,70,85,50,90].map((w,i)=><div key={i} className="shimmer shimmer-line" style={{width:`${w}%`,animationDelay:`${i*.1}s`}}/>)}</div>}
      {result && !loading && (
        <div className="out-section">
          <div className="out-grid">
            <div className="out-card accent">
              <div className="out-label">Recommended Model</div>
              <div className="out-title">{result.model?.name}</div>
              <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:".68rem",color:"var(--green)",margin:"4px 0 8px"}}>{result.model?.architecture}</div>
              <div className="out-body">{result.model?.reason}</div>
            </div>
            <div className="out-card">
              <div className="out-label">Datasets</div>
              {[result.dataset,result.dataset2].filter(Boolean).map(ds=>(
                <div key={ds.name} style={{marginBottom:10}}>
                  <div style={{fontWeight:600,fontSize:".86rem",marginBottom:3}}>{ds.name}</div>
                  <div className="out-body" style={{fontSize:".75rem",marginBottom:4}}>{ds.size}{ds.note?` — ${ds.note}`:""}</div>
                  {ds.url&&ds.url!=="string"&&<a href={ds.url} target="_blank" rel="noreferrer" className="out-link">Open Dataset ↗</a>}
                </div>
              ))}
            </div>
          </div>
          <div className="out-card" style={{marginTop:12,background:"var(--s2)",border:"1px solid var(--border)",borderRadius:10,padding:18}}>
            <div className="out-label">Evaluation Metrics</div>
            {result.metrics?.map(m=>(
              <div key={m.name} className="metric-row"><div className="metric-name">{m.name}</div><div className="metric-val">{m.target}</div><div className="metric-desc">{m.why}</div></div>
            ))}
          </div>
          <div className="out-grid" style={{marginTop:12}}>
            <div className="out-card">
              <div className="out-label">Key Challenges</div>
              {result.challenges?.map((c,i)=><div key={i} className="challenge-item"><div className="challenge-num">0{i+1}</div><div style={{fontSize:".82rem"}}>{c}</div></div>)}
              {result.clinicalStake&&<div className="stake-box" style={{marginTop:12}}><div style={{fontSize:"1.1rem"}}>🚨</div><div className="stake-text">{result.clinicalStake}</div></div>}
            </div>
            <div className="out-card">
              <div className="out-label">Tips for {level==="beginner"?"Beginners":level==="intermediate"?"Intermediate Devs":"Advanced Researchers"}</div>
              {result.tips?.map((t,i)=><div key={i} style={{display:"flex",gap:10,padding:"10px 0",borderBottom:"1px solid var(--border)"}}><div style={{color:"var(--green)"}}>✓</div><div style={{fontSize:".83rem",lineHeight:1.6}}>{t}</div></div>)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Blueprint() {
  const [topic, setTopic] = useState("");
  const [purpose, setPurpose] = useState("science-fair");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [saved, setSaved] = useState(false);
  const EXAMPLES = ["AI for early Parkinson's detection using voice","Predicting sepsis risk from ICU vitals","Classifying diabetic retinopathy stages","Detecting rare diseases from facial photos"];

  async function generate() {
    if (!topic.trim()) return;
    setLoading(true); setResult(null); setErr(""); setSaved(false);
    try {
      const text = await callAI(`You are a research mentor for a high school student creating a ${purpose} blueprint. Topic: "${topic}".
Return ONLY valid JSON (no markdown fences):
{"title":"string","question":"string","hypothesis":"string","impact":"string","datasets":[{"name":"string","url":"string","size":"string"},{"name":"string","url":"string","size":"string"}],"model":{"name":"string","reason":"string"},"timeline":[{"week":"Weeks 1-2","task":"string"},{"week":"Weeks 3-4","task":"string"},{"week":"Weeks 5-6","task":"string"},{"week":"Weeks 7-8","task":"string"}],"metrics":["string","string","string"],"novelty":"string"}`);
      const parsed = extractJSON(text);
      if (!parsed) throw new Error("Parse error — please try again.");
      setResult(parsed);
    } catch(e) { setErr(e.message); }
    setLoading(false);
  }

  function save() {
    if (!result) return;
    try {
      localStorage.setItem(`bp_${Date.now()}`, JSON.stringify({topic, purpose, result, date: new Date().toLocaleDateString()}));
      setSaved(true);
    } catch(e) { console.error(e); }
  }

  return (
    <div>
      <div className="card">
        <div className="card-head">📐 Research Blueprint Generator</div>
        <div className="card-sub">Turn any AI healthcare idea into a complete research plan — question, hypothesis, datasets, timeline, and metrics.</div>
        <div className="chips">{EXAMPLES.map(e=><div key={e} className={`chip ${topic===e?"active":""}`} onClick={()=>setTopic(e)}>{e}</div>)}</div>
        <div className="field"><label>Research Topic</label><input placeholder="Describe your AI healthcare research idea..." value={topic} onChange={e=>setTopic(e.target.value)}/></div>
        <div className="form-row">
          <div className="field"><label>Purpose</label>
            <select value={purpose} onChange={e=>setPurpose(e.target.value)}>
              <option value="science-fair">Science Fair (ISEF, Regeneron)</option>
              <option value="hackathon">Hackathon</option>
              <option value="research-paper">Research Paper / Journal</option>
              <option value="class-project">Class Project</option>
              <option value="startup">Startup / Demo Day</option>
            </select>
          </div>
        </div>
        {err && <div style={{color:"var(--red)",fontSize:".8rem",marginBottom:12}}>{err}</div>}
        <button className="btn btn-green" onClick={generate} disabled={loading||!topic.trim()}>
          {loading?<><div className="spinner"/>Generating...</>:"Generate Blueprint →"}
        </button>
      </div>
      {loading && <div className="card">{[100,60,80,45,95,55].map((w,i)=><div key={i} className="shimmer shimmer-line" style={{width:`${w}%`,animationDelay:`${i*.1}s`}}/>)}</div>}
      {result && !loading && (
        <div className="out-section">
          <div className="card" style={{borderColor:"rgba(0,255,157,.3)",background:"rgba(0,255,157,.03)"}}>
            <div style={{display:"flex",alignItems:"flex-start",justifyContent:"space-between",gap:12,flexWrap:"wrap"}}>
              <div>
                <div className="out-label">Project Title</div>
                <div style={{fontFamily:"'Barlow Condensed',sans-serif",fontSize:"1.5rem",fontWeight:800,marginBottom:6}}>{result.title}</div>
              </div>
              <button className="btn btn-ghost" onClick={save} disabled={saved}>{saved?"✓ Saved":"💾 Save"}</button>
            </div>
            <div style={{marginTop:14,display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
              <div className="info-panel"><div className="info-panel-label">Research Question</div><p>{result.question}</p></div>
              <div className="info-panel"><div className="info-panel-label">Hypothesis</div><p>{result.hypothesis}</p></div>
              <div className="info-panel full" style={{gridColumn:"1/-1"}}><div className="info-panel-label">Real-World Impact</div><p>{result.impact}</p></div>
            </div>
          </div>
          <div className="out-grid">
            <div className="out-card accent">
              <div className="out-label">Suggested Model</div>
              <div className="out-title">{result.model?.name}</div>
              <div className="out-body" style={{marginTop:6}}>{result.model?.reason}</div>
            </div>
            <div className="out-card">
              <div className="out-label">Datasets</div>
              {result.datasets?.map(ds=>(
                <div key={ds.name} style={{marginBottom:10}}>
                  <div style={{fontWeight:600,fontSize:".84rem"}}>{ds.name}</div>
                  <div className="out-body" style={{fontSize:".74rem",marginBottom:4}}>{ds.size}</div>
                  {ds.url&&ds.url!=="string"&&<a href={ds.url} target="_blank" rel="noreferrer" className="out-link">Open ↗</a>}
                </div>
              ))}
            </div>
          </div>
          <div style={{background:"var(--s2)",border:"1px solid var(--border)",borderRadius:10,padding:20,marginTop:12}}>
            <div className="out-label">Project Timeline</div>
            <div className="timeline">
              {result.timeline?.map((t,i)=>(
                <div key={i} className="tl-item">
                  <div className="tl-dot">{i+1}</div>
                  <div><div className="tl-week">{t.week}</div><div className="tl-task">{t.task}</div></div>
                </div>
              ))}
            </div>
          </div>
          <div className="out-grid" style={{marginTop:12}}>
            <div className="out-card">
              <div className="out-label">Success Metrics</div>
              {result.metrics?.map((m,i)=><div key={i} style={{display:"flex",gap:8,padding:"8px 0",borderBottom:"1px solid var(--border)"}}><div style={{color:"var(--green)"}}>→</div><div style={{fontSize:".83rem"}}>{m}</div></div>)}
            </div>
            <div className="out-card accent">
              <div className="out-label">What Makes This Novel</div>
              <div className="out-body" style={{fontSize:".85rem",lineHeight:1.7}}>{result.novelty}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Saved() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(null);

  useEffect(()=>{
    try {
      const keys = Object.keys(localStorage).filter(k=>k.startsWith("bp_"));
      const loaded = keys.map(k=>{
        try { return {key:k,...JSON.parse(localStorage.getItem(k))}; } catch { return null; }
      }).filter(Boolean).reverse();
      setItems(loaded);
    } catch(e) { console.error(e); }
    setLoading(false);
  },[]);

  function remove(key) {
    try { localStorage.removeItem(key); setItems(items.filter(i=>i.key!==key)); } catch {}
  }

  if (loading) return <div className="card"><div className="shimmer shimmer-line" style={{width:"60%"}}/></div>;
  if (!items.length) return (
    <div className="card" style={{textAlign:"center",padding:48}}>
      <div style={{fontSize:"2rem",marginBottom:12}}>📂</div>
      <div style={{fontFamily:"'Barlow Condensed',sans-serif",fontSize:"1.2rem",fontWeight:700,marginBottom:8}}>No saved blueprints yet</div>
      <div style={{color:"var(--muted)",fontSize:".85rem"}}>Generate a blueprint and hit Save.</div>
    </div>
  );

  return (
    <div className="card">
      <div className="card-head">💾 Saved Blueprints</div>
      <div className="card-sub">{items.length} blueprint{items.length!==1?"s":""} saved</div>
      {items.map(item=>(
        <div key={item.key}>
          <div className="saved-card" onClick={()=>setExpanded(expanded===item.key?null:item.key)}>
            <div>
              <div style={{fontFamily:"'Barlow Condensed',sans-serif",fontWeight:700,fontSize:"1rem"}}>{item.result?.title||item.topic}</div>
              <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:".65rem",color:"var(--muted)",marginTop:3}}>{item.purpose?.replace("-"," ")} · {item.date}</div>
            </div>
            <div style={{display:"flex",gap:8,alignItems:"center"}}>
              <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:".65rem",color:"var(--muted)"}}>{expanded===item.key?"▲":"▼"}</span>
              <button className="btn btn-ghost" style={{padding:"5px 10px",fontSize:".65rem"}} onClick={e=>{e.stopPropagation();remove(item.key);}}>Delete</button>
            </div>
          </div>
          {expanded===item.key&&item.result&&(
            <div className="card" style={{marginTop:4,marginBottom:4,borderColor:"rgba(0,255,157,.2)"}}>
              <div style={{fontSize:".85rem",color:"var(--muted)",marginBottom:10}}><strong style={{color:"var(--text)"}}>Q: </strong>{item.result.question}</div>
              <div style={{fontSize:".85rem",color:"var(--muted)",marginBottom:10}}><strong style={{color:"var(--text)"}}>H: </strong>{item.result.hypothesis}</div>
              <div style={{fontSize:".85rem",color:"var(--muted)"}}><strong style={{color:"var(--text)"}}>Impact: </strong>{item.result.impact}</div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
