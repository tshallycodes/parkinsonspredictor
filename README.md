# 🫀 Parkinson's Finger Tapping Project
This project develops a computer-based assessment tool that analyses finger tapping movements to detect Parkinson's Disease. Raw amplitude and time signals are processed and fed into a machine learning pipeline to classify patients as Parkinson's positive or healthy controls. This repository contains all relevant files for the project.

## 📌 Project Overview
Parkinson's Disease affects motor function, often impairing fine motor skills such as finger tapping. This project leverages signal processing and machine learning to:
- Process raw finger tapping amplitude & time signal data
- Extract meaningful statistical and signal-based features
- Classify subjects as:
  - `1` → Parkinson's Disease
  - `0` → Healthy Control
- Evaluate model performance using robust cross-validation techniques
- Export trained models for deployment via a Streamlit web app

## 📊 Dataset Description
Each patient folder contains:
- `Amplitude.txt` → Finger tapping amplitude values
- `Time.txt` → Corresponding timestamps

Additionally:
- `updrs_scores.csv` → Clinical motor severity scores (UPDRS), used as a feature

### Target Variable
The label is derived from the patient ID:
- `PD*` → Parkinson's Disease (`1`)
- Otherwise → Healthy Control (`0`)

### Hand Separation
Data is split by hand and modelled independently:
- `left_features_df` → Left hand signals
- `right_features_df` → Right hand signals

This preserves side-specific motor patterns, as Parkinson's symptoms can be asymmetric.

## ⚙️ Pipeline Overview

### 1️⃣ Data Retrieval
- Loads data from Google Drive
- Reads amplitude and time signals per patient
- Loads and merges UPDRS scores

### 2️⃣ Data Preprocessing
- Pads signals to a fixed length (1800 samples)
- Converts raw lists to NumPy arrays
- Handles missing or inconsistent files

### 3️⃣ Feature Engineering
Signal-based feature extraction includes:
- Statistical features: mean, standard deviation, variance, min/max
- Signal smoothing using Savitzky–Golay filter
- Peak detection using `scipy.signal.find_peaks`
- Frequency-based and amplitude-based derived metrics
- UPDRS score (encoded as binary feature: `1` if score ≥ 3, else `0`)

### 4️⃣ Model Training & Evaluation
**Model trained:**
- SVM (RBF Kernel, `C=0.5`, `class_weight='balanced'`)
- Trained separately for left and right hand

**Validation strategy:**
- Stratified K-Fold Cross-Validation (5 splits)
- Cross-validated predictions via `cross_val_predict`

**Metrics reported:**
- Accuracy
- Precision
- Recall
- F1 Score
- Weighted F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- Uncertainty Rate (predictions in the 0.40–0.55 probability band)

## 🚀 Deployment
Trained models are saved as `.pkl` files and served via a **Streamlit web app**:
- User selects hand (Left / Right) and inputs UPDRS score
- Paste amplitude and time signal values
- App returns prediction with confidence probability
```
model = joblib.load("models/left_svm.pkl")
prediction = model.predict_proba(features)
```

## 🧠 Why This Matters
Early detection of Parkinson's Disease can:
- Improve treatment planning and medication timing
- Support clinical decision-making with objective motor data
- Enable remote motor function assessment without specialist equipment

This pipeline demonstrates how signal processing and machine learning can support neurological diagnosis in a lightweight, deployable format.

## 🔮 Future Improvements
- Deep learning on raw signals (LSTM / CNN)
- Frequency-domain feature extraction (FFT)
- Feature selection & dimensionality reduction (PCA)
- Hyperparameter tuning (GridSearchCV)
- Class imbalance handling (SMOTE)
- External validation dataset
- Multi-class UPDRS severity prediction (extending beyond binary classification)

## 📜 License
This project is for academic and research purposes.