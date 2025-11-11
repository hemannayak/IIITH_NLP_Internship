# IIITH_NLP_Internship

# VoiceScope India: AI-Powered Native Language Identification & Cultural Cuisine Discovery

**Project Title:** VoiceScope India: AI-Powered Native Language Identification & Cultural Cuisine Discovery

**Student Name(s):** Pangoth Hemanth Nayak, Arutla Prasanna, Apurba Nandi

**Roll Number(s):** 23E51A67C5, 23E51A6711, 23e51A6708

**Department & Institution:** Department of Computer Science And Engineering - Data Science & Hyderabad Institute of Technology and Management (HITAM)


---

## 1. Project Scope and Deliverables

### 1.1 Project Scope
The main goal of this project is to develop an AI system capable of identifying the native language (accent) of a speaker from their English speech. The system focuses on Indian accents from seven different languages: Bengali, Malayalam, Telugu, Hindi, Marathi, Kannada, and Tamil. A key aspect of the project is to evaluate the model's ability to generalize its accent detection capabilities from adult speech (used for training and validation) to child speech (used for testing), addressing the challenge of cross-age generalization. The problem it solves is providing insights into a speaker's linguistic background based on their accent, which can have applications in personalized user experiences, content recommendation, and linguistic research.

### 1.2 Deliverables
The deliverables for this project include:
-   Processed and annotated dataset (including age and speech type metadata).
-   Acoustic feature extraction pipeline (MFCC and HuBERT embeddings).
-   A trained deep neural network model for accent classification.
-   Evaluation of the model's performance, including cross-age generalization analysis.
-   Analysis of performance based on speech type (word vs. sentence level).
-   Analysis of the effectiveness of different HuBERT layers for accent detection.
-   A functional Streamlit web application demonstrating real-time accent detection and a novel accent-aware cuisine recommendation system.
-   Quantized model weights for efficient deployment.

## 2. Model Development

### 2.1 Feature Extraction
A combination of traditional and state-of-the-art acoustic features is used to represent the audio:
-   **MFCC Features:** Mel-Frequency Cepstral Coefficients (MFCCs) capture the spectral envelope of the audio, which is known to be relevant for speaker and accent characteristics. We extract 40 MFCCs per frame and pad/truncate sequences to a maximum length determined from the training data (`MAX_MFCC_LEN = 313` frames).
-   **HuBERT Embeddings:** Hidden-Unit Bidirectional Encoder Representations from Transformer (HuBERT) is a self-supervised model pre-trained on large amounts of speech. It captures high-level, abstract representations of speech. We extract the mean-pooled embeddings from a specific hidden layer (Layer 9 by default, based on common practice for speaker/accent tasks, and further analyzed in Section 5).

These features are concatenated to form a combined feature vector for each audio sample, with a total dimension of 13288 features (12520 from MFCC + 768 from HuBERT).

### 2.2 Model Architectures
The model used for accent classification is a Deep Neural Network (DNN), specifically a 4-layer Multilayer Perceptron (MLP).
-   Input Layer: 13288 features (combined MFCC + HuBERT).
-   Hidden Layers: Three fully connected layers with Batch Normalization and ReLU activation, and Dropout (p=0.4) for regularization. The hidden dimensions are 512, 256, and 128.
-   Output Layer: A final fully connected layer with 7 output nodes, corresponding to the 7 target accent classes (languages).

### 2.3 Training and Optimization
-   **Dataset Split:** The dataset is split based on age. Adult samples are used for training (80%) and validation (20%), stratified by language label to maintain class distribution. Child samples are used exclusively as a separate test set for evaluating cross-age generalization.
-   **Training Process:** The model is trained for 40 epochs using the AdamW optimizer. Cross-Entropy Loss is used as the criterion.
-   **Optimization:** A ReduceLROnPlateau scheduler is used to reduce the learning rate when the validation accuracy plateaus, helping the model converge effectively.
-   **Validation:** The model's performance is monitored on the validation set after each epoch. The model with the best validation accuracy is saved.

## 3. Generalization Across Age Groups

A core aspect of this project is evaluating how well the model trained on adult speech generalizes to child speech.

-   **Training data (Adults):** The model was trained and validated exclusively on speech samples from adult speakers (18-35 years) from the IIIT dataset, covering all 7 languages.
-   **Testing data (Children):** The model's generalization capability was tested on a separate set consisting solely of child speech samples from the dataset.

Based on the evaluation (Cell `jOan0a5NCBQa`), the model achieved **100.00% accuracy** on the child test set. This indicates excellent generalization from adult-trained data to child voices within this specific dataset. The cross-age generalization gap was 0.00%.

## 4. Word-Level vs Sentence-Level Accent Detection

Analysis was conducted to understand if the model's performance differs between word-level and sentence-level speech.

Based on the dataset structure (Cell `ZYprjD-28rXL`) and subsequent analysis (Cell `faK_kecXCIZa`), all samples in the IIIT dataset used were identified as 'sentence'-level speech. Therefore, a direct comparison of model performance between distinct 'word'-level and 'sentence'-level samples from this dataset was not possible in this experiment.

| Comparison Criteria | Word-Level | Sentence-Level |
|---------------------|------------|----------------|
| Accuracy            | N/A        | 100.00% (Val), 100.00% (Test) |
| Robustness          | N/A        | [Observations based on future testing if word data is available] |
| Interpretability    | N/A        | [Observations based on future testing if word data is available] |

*(Note: Further analysis on this aspect would require a dataset containing distinct word-level and sentence-level recordings for the same speakers/accents.)*

## 5. HuBERT Layer-wise Analysis

To understand which layer of the pre-trained HuBERT model is most informative for accent detection, an analysis was performed by extracting embeddings from different layers and evaluating the model's performance.

The analysis (Cell `PUY4y07rCKh3`) tested layers 3, 6, 9, and 11 using a subset of the validation data.

| HuBERT Layer | Validation Accuracy (%) |
|--------------|-------------------------|
| 3            | 99.80%                  |
| 6            | 100.00%                 |
| 9            | 100.00%                 |
| 11           | 100.00%                 |

Layers 6, 9, and 11 all achieved 100.00% accuracy on the validation subset, indicating that the middle to later layers of HuBERT capture highly effective accent cues for this dataset. Layer 6 was identified as the best performing layer based on this specific test, although layers 9 and 11 performed equally well on this subset. Layer 9 was used for the main feature extraction pipeline based on common research findings, but layer 6 could also be a strong candidate.

## 6. Results and Discussion

The model achieved high accuracy on both the adult validation set and the child test set, demonstrating excellent cross-age generalization within the scope of the IIIT dataset.

| Experiment                   | Model             | Feature       | Accuracy   | Key Observation                                  |
|------------------------------|-------------------|---------------|------------|--------------------------------------------------|
| Accent Classification (Adults) | 4-layer MLP (DNN) | MFCC + HuBERT | 100.00%    | Model effectively learns accent patterns from adults. |
| Cross-Age Generalization     | 4-layer MLP (DNN) | MFCC + HuBERT | 100.00%    | Excellent generalization to child voices on test set. |

The perfect accuracy on the test set, while promising, should be interpreted considering the limited diversity of accents (Hindi and Tamil) in the child test split. Future work could involve testing on a more diverse and potentially noisy dataset to fully assess real-world robustness. The combination of MFCC and HuBERT features appears to be a powerful approach for this task.

## 7. Application Development

### Accent-Aware Cuisine Recommendation System

The developed accent detection model is integrated into a Streamlit web application (`app.py`) to create an interactive **Accent-Aware Cuisine Recommendation System**.

**How it works:**
1.  The user interacts with the Streamlit app (accessible via the ngrok tunnel or deployed on Streamlit Cloud).
2.  The user can either upload an audio file (WAV, MP3, M4A) or record their voice directly in the browser.
3.  The application preprocesses the audio, extracts the combined MFCC and HuBERT features, and feeds them into the trained accent classification model.
4.  The model predicts the speaker's native language (accent) and provides a confidence score.
5.  Based on the predicted accent, the application provides personalized cuisine recommendations associated with the detected linguistic region of India, along with brief information about that cuisine.

**What it demonstrates:**
This application demonstrates a practical use case for accent detection technology. By linking linguistic identity to cultural aspects like cuisine, it provides a user-friendly and engaging experience. It showcases the end-to-end pipeline, from feature extraction and model inference to a deployed, interactive application. The cross-age generalization capability is implicitly demonstrated if the app can correctly identify accents from both adult and child users.

| Detected Accent | Inferred Region     | Recommended Dishes                     |
|-----------------|---------------------|------------------------------------|
| Bengali         | East India 🌊       | Hilsa Fish Curry, Mishti Doi, etc. |
| Malayalam       | South India 🌴      | Appam & Stew, Malabar Fish Curry, etc. |
| Telugu          | South India 🌴      | Hyderabadi Biryani, Gongura Chicken, etc. |
| Hindi           | North India 🏔️      | Butter Chicken, Chole Bhature, etc. |
| Marathi         | West India 🏜️       | Vada Pav, Misal Pav, etc.          |
| Kannada         | South India 🌴      | Bisi Bele Bath, Mysore Dosa, etc. |
| Tamil           | South India 🌴      | Chettinad Chicken, Idli-Sambar, etc. |

## 8. Tools and Frameworks Used
-   **Programming Language:** Python
-   **Libraries:**
    -   PyTorch (for building and training the neural network)
    -   Transformers (for utilizing the HuBERT model)
    -   Librosa (for audio loading and MFCC feature extraction)
    -   Datasets (for dataset handling)
    -   Scikit-learn (for data splitting, encoding, and evaluation metrics)
    -   NumPy (for numerical operations)
    -   Pandas (for data manipulation and analysis)
    -   Matplotlib & Seaborn (for visualizations)
    -   Tqdm (for progress bars)
    -   Joblib (for saving/loading scaler and encoder)
    -   Streamlit (for building the web application)
    -   audio-recorder-streamlit & streamlit-audiorec (for audio recording in Streamlit)
    -   pyngrok (for creating a public URL for the Streamlit app)
    -   Soundfile (for handling audio file formats)
-   **Tools:** Google Colab (for development and execution environment), Google Drive (for dataset storage), Streamlit Cloud (for deployment).

## 9. Conclusion
This project successfully developed an AI system capable of classifying Indian English accents from seven different native language backgrounds. A key finding is the excellent cross-age generalization achieved by training the model on adult speech and testing on child speech, demonstrating the robustness of the chosen features and model architecture for this specific dataset. The combination of traditional MFCC features and advanced HuBERT embeddings proved highly effective. The project culminates in a functional Streamlit application, showcasing a practical application of accent detection for personalized cultural recommendations.

## 10. Future Work
-   Evaluate the model on a more diverse and larger dataset of child speech, including all 7 languages, to further validate cross-age generalization.
-   Test the model's robustness to various types and levels of background noise and different recording environments.
-   Explore data augmentation techniques, such as pitch shifting or adding noise, to improve generalization and robustness.
-   Investigate the use of other self-supervised models (e.g., Wav2Vec 2.0, XLS-R) or fine-tuning the HuBERT model on accent data.
-   Expand the dataset to include more Indian languages and accents.
-   Develop a more sophisticated recommendation system that considers user preferences beyond just cuisine.
-   Optimize the model further for edge device deployment.

## 11. Code Links
The code for this project is available in the following Google Colab notebook:
[Insert Link to your Google Colab Notebook Here]

For easier deployment and version control, we recommend uploading the following files to a GitHub repository with proper documentation:
-   `app.py` (Streamlit application code)
-   `accent_model.pt` (Quantized PyTorch model weights)
-   `scaler.pkl` (StandardScaler object)
-   `label_encoder.pkl` (LabelEncoder object)
-   `requirements.txt` (List of Python package dependencies - see below)

**`requirements.txt` content:**
