
# VoiceScope India  
### AI-Powered Native Language Identification & Cultural Cuisine Discovery

**Students:** Pangoth Hemanth Nayak, Arutla Prasanna, Apurba Nandi  
**Roll Numbers:** 23E51A67C5, 23E51A6711, 23E51A6708  
**Department:** CSE (Data Science), HITAM  

---

## 📌 Overview

**VoiceScope India** is an AI-driven system that identifies the **native accent** of Indian speakers from their **English speech**, covering six major languages:

**Telugu, Tamil, Malayalam, Kannada, Hindi, Gujarati**

The project integrates:

- MFCC-based acoustic features  
- HuBERT self-supervised speech embeddings  
- A robust Deep Neural Network  
- A Streamlit web app with **Accent-Aware Cuisine Recommendations**

This project also evaluates the **cross-age generalization** capability by training on **adult speech** and testing on **child speech**.

---

## 🎯 Project Scope

This project aims to:

- Detect the Indian native accent from English speech using AI.  
- Compare generalization performance between adult-trained and child-tested speech.  
- Build a combined MFCC + HuBERT feature pipeline.  
- Deploy a Streamlit-based interactive web app.  
- Provide region-based cuisine suggestions based on detected accent.

---

## 📦 Deliverables

- Preprocessed and annotated dataset metadata  
- MFCC + HuBERT feature extraction modules  
- 4-layer DNN model for accent classification  
- Training, validation, and testing scripts  
- Cross-age generalization analysis  
- HuBERT layer-wise performance evaluation  
- Streamlit application (`app.py`)  
- Quantized PyTorch model weights  
- Scaler and label encoder utilities  
- Deployment-ready `requirements.txt`

---

## 🧠 Model Development

### 🔹 Feature Extraction

| Feature Type | Description |
|--------------|-------------|
| **MFCC (40 coefficients)** | Captures spectral representation of speech |
| **HuBERT Embedding (Layer 9)** | High-level self-supervised speech features |

Final feature vector length = **13,288**  
(12,520 MFCC + 768 HuBERT)

---

### 🔹 Model Architecture

A 4-layer **Multilayer Perceptron**:

- Input: **13,288 features**
- Hidden Layers: `512 → 256 → 128`  
  (BatchNorm + ReLU + Dropout=0.4)
- Output: **6 accent classes**

**Training Configuration:**

- Optimizer: AdamW  
- Loss: CrossEntropy  
- Epochs: 40  
- LR Scheduler: ReduceLROnPlateau  

---

## 👶 Cross-Age Generalization

| Dataset | Purpose |
|---------|---------|
| Adults | Training + Validation |
| Children | Final Testing |

- **Adult Validation Accuracy:** 99.87%  
- **Child Test Accuracy:** 91.91%  
- **Generalization Gap:** ~7.96%

The model generalizes well, despite acoustic differences between adult and child voices.

---

## 🗣️ Word vs Sentence-Level Performance

The IIIT dataset includes **only sentence-level** samples.  
Therefore, a word-vs-sentence comparison was not possible.

---

## 🔬 HuBERT Layer-wise Analysis

| HuBERT Layer | Accuracy (%) |
|--------------|--------------|
| Layer 3 | 98.00 |
| Layer 6 | 98.80 |
| **Layer 9** | **99.60** |
| Layer 11 | 99.40 |

Layer 9 was the best-performing layer and used for the final model.

---

## 📊 Results Summary

| Task | Accuracy | Insight |
|------|----------|---------|
| Adult Validation | **99.87%** | Excellent accent separation |
| Child Test | **91.91%** | Strong cross-age generalization |

---

## 🖥️ Streamlit Web Application

The web app demonstrates:

1. Real-time audio upload/recording  
2. MFCC + HuBERT feature extraction  
3. Accent prediction with confidence scoring  
4. Cuisine recommendations based on predicted accent  

### 🍽️ Accent-Aware Cuisine Mapping

| Accent | Region | Popular Dishes |
|--------|--------|----------------|
| Gujarati | West India | Dhokla, Thepla, Undhiyu |
| Hindi | North India | Chole Bhature, Butter Chicken |
| Kannada | South India | Bisi Bele Bath, Mysore Dosa |
| Malayalam | South India | Appam & Stew, Malabar Curry |
| Tamil | South India | Idli-Sambar, Chettinad Chicken |
| Telugu | South India | Hyderabadi Biryani, Gongura Chicken |

---

## 🛠️ Tech Stack

### Languages
- Python

### Libraries/Frameworks Used
- PyTorch  
- Transformers  
- Librosa  
- Streamlit  
- Scikit-learn  
- NumPy, Pandas  
- Soundfile  
- audio-recorder-streamlit, streamlit-audiorec  
- Matplotlib, Seaborn  

### Tools
- Google Colab  
- Google Drive  
- Streamlit Cloud  
- pyngrok  

---

## 🚀 Future Work

- Add robust noise-tolerant training  
- Data augmentation (pitch shift, noise, speed scaling)  
- Large-scale child speech dataset evaluation  
- Expand coverage to 14+ Indian languages  
- Try fine-tuned HuBERT / Wav2Vec / XLS-R models  
- Build multi-domain cultural recommendations  
- Optimize for mobile/edge deployment  

---


## 🏁 Conclusion

VoiceScope India successfully demonstrates:

- Reliable Indian accent classification  
- Strong cross-age generalization  
- Effective combination of MFCC + HuBERT embeddings  
- A complete end-to-end application via Streamlit  
- Cultural personalization through cuisine recommendations  

This makes it a practical, deployable, and extensible AI solution at the intersection of speech technology and cultural intelligence.

---

