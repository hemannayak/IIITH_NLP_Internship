
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import joblib
import tempfile
import os
from audio_recorder_streamlit import audio_recorder
import io
import soundfile as sf
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# ============= PAGE CONFIGURATION =============
st.set_page_config(
    page_title="VoiceScope India | AI Accent Classifier",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============= CUSTOM CSS =============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    animation: bgflow 20s ease infinite;
    background-size: 400% 400%;
}

@keyframes bgflow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.hero-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    padding: 3.5rem 2rem;
    border-radius: 30px;
    text-align: center;
    color: white;
    margin-bottom: 2.5rem;
    box-shadow: 0 25px 70px rgba(0,0,0,0.35);
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: rotate 30s linear infinite;
}

@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.hero-header h1 {
    font-size: 3.5rem;
    margin: 0;
    font-weight: 700;
    position: relative;
    z-index: 1;
    text-shadow: 0 4px 15px rgba(0,0,0,0.4);
}

.hero-subtitle {
    font-size: 1.3rem;
    margin-top: 1rem;
    opacity: 0.95;
    position: relative;
    z-index: 1;
}

.glass-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(25px);
    border-radius: 25px;
    padding: 2.5rem;
    margin: 2rem 0;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 15px 50px rgba(0,0,0,0.3);
    transition: transform 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
}

.result-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 3.5rem 2rem;
    border-radius: 30px;
    color: white;
    text-align: center;
    box-shadow: 0 25px 60px rgba(0,0,0,0.4);
    margin: 2.5rem 0;
    animation: slideUp 0.8s ease-out;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(50px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result-box h2 {
    font-size: 3rem;
    margin-bottom: 1rem;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.05);
    }
}

.confidence-bar-container {
    background: rgba(255, 255, 255, 0.25);
    height: 25px;
    border-radius: 15px;
    margin: 1.5rem auto;
    max_width: 500px;
    overflow: hidden;
    box-shadow: inset 0 0 15px rgba(0,0,0,0.2);
}

.confidence-bar {
    height: 100%;
    background: linear-gradient(90deg, #fff, #ffd700, #fff);
    animation: fillBar 2s ease-out forwards, shimmer 3s ease-in-out infinite;
    border-radius: 15px;
}

@keyframes fillBar {
    from { width: 0%; }
}

@keyframes shimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.cuisine-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 1.5rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 20px 50px rgba(102, 126, 234, 0.5);
    transition: all 0.4s ease;
    margin: 0.5rem;
}

.cuisine-card:hover {
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0 25px 70px rgba(118, 75, 162, 0.7);
}

.info-section {
    background: linear-gradient(135deg, #434343 0%, #000000 100%);
    padding: 3rem 2rem;
    border-radius: 25px;
    color: white;
    margin-top: 3rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}

.stat-box {
    background: rgba(255, 255, 255, 0.1);
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    margin: 1rem 0;
}

.team-section {
    background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
    padding: 3rem 2rem;
    border-radius: 25px;
    color: white;
    text-align: center;
    margin-top: 3rem;
}

.team-member {
    display: inline-block;
    background: rgba(255, 255, 255, 0.15);
    padding: 1rem 2rem;
    margin: 0.5rem;
    border-radius: 25px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    transition: all 0.4s ease;
}

.team-member:hover {
    transform: scale(1.1);
    border-color: #ffd700;
    background: rgba(255, 255, 255, 0.25);
}

#MainMenu, footer {
    visibility: hidden;
}

.stRadio > div {
    background: rgba(255, 255, 255, 0.1);
    padding: 1rem;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ============= HERO HEADER =============
st.markdown("""
<div class="hero-header">
    <h1>🎤 VoiceScope India</h1>
    <p class="hero-subtitle">AI-Powered Native Language Identification & Cultural Cuisine Discovery</p>
    <p class="hero-subtitle" style="font-size:1rem; margin-top:0.5rem;">
        Cross-Age Generalization | HuBERT + MFCC | Real-time Accent Detection
    </p>
</div>
""", unsafe_allow_html=True)

# ============= MODEL ARCHITECTURE =============
class AccentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)

# ============= LOAD MODEL & RESOURCES =============
@st.cache_resource
def load_model_resources():
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    # Removed max_mfcc_len from here to be accessed via scaler directly
    input_dim = scaler.n_features_in_
    num_classes = len(label_encoder.classes_)

    # 1. Initialize the non-quantized model
    non_quantized_model = AccentClassifier(input_dim, 512, num_classes)

    # 2. Dynamically quantize the model
    model = torch.quantization.quantize_dynamic(
        non_quantized_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # 3. Load the state_dict from the saved quantized model
    model.load_state_dict(torch.load('accent_model.pt', map_location='cpu'))
    model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert.eval()

    return model, scaler, label_encoder, feature_extractor, hubert

# ============= CUISINE MAPPING =============
region_cuisine_map = {
    'Bengali': {
        'region': 'East India 🌊',
        'state': 'West Bengal',
        'cuisines': ['🐟 Hilsa Fish Curry', '🍚 Litti-Chokha', '🍮 Mishti Doi', '🍤 Chingri Malai'],
        'icon': '🌊',
        'description': 'Bengali cuisine is known for its subtle flavors and emphasis on fish and rice.'
    },
    'Malayalam': {
        'region': 'South India 🌴',
        'state': 'Kerala',
        'cuisines': ['🥞 Appam & Stew', '🐟 Malabar Fish Curry', '🌾 Puttu & Kadala', '🥥 Avial'],
        'icon': '🌴',
        'description': 'Kerala cuisine features coconut, seafood, and aromatic spices.'
    },
    'Telugu': {
        'region': 'South India 🌴',
        'state': 'Andhra Pradesh',
        'cuisines': ['🍛 Hyderabadi Biryani', '🌶️ Gongura Chicken', '🥞 Pesarattu', '🍖 Andhra Curry'],
        'icon': '🌴',
        'description': 'Telugu cuisine is known for its spicy and tangy flavors.'
    },
    'Hindi': {
        'region': 'North India 🏔️',
        'state': 'Multiple States',
        'cuisines': ['🍗 Butter Chicken', '🫘 Dal Makhani', '🫓 Chole Bhature', '🍛 Rajma Chawal'],
        'icon': '🏔️',
        'description': 'North Indian cuisine features rich gravies and tandoori preparations.'
    },
    'Marathi': {
        'region': 'West India 🏜️',
        'state': 'Maharashtra',
        'cuisines': ['🍔 Vada Pav', '🥞 Puran Poli', '🌶️ Misal Pav', '🐟 Bombil Fry'],
        'icon': '🏜️',
        'description': 'Maharashtrian cuisine balances spicy, sweet, and tangy flavors.'
    },
    'Kannada': {
        'region': 'South India 🌴',
        'state': 'Karnataka',
        'cuisines': ['🍛 Bisi Bele Bath', '🥞 Mysore Dosa', '🌾 Ragi Mudde', '🥘 Coorg Pork'],
        'icon': '🌴',
        'description': 'Karnataka cuisine offers diverse flavors from coastal to interior regions.'
    },
    'Tamil': {
        'region': 'South India 🌴',
        'state': 'Tamil Nadu',
        'cuisines': ['🍗 Chettinad Chicken', '🥞 Idli-Sambar', '🍛 Pongal', '🐟 Meen Kuzhambu'],
        'icon': '🌴',
        'description': 'Tamil cuisine emphasizes rice, lentils, and aromatic spice blends.'
    }
}

# These are global variables
model, scaler, label_encoder, feature_extractor, hubert = load_model_resources()

# ============= FEATURE EXTRACTION =============
def extract_mfcc(wav, sr, n_mfcc=40, max_len=None):
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc, hop_length=512, n_fft=2048)
    if max_len:
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
    return mfcc

def extract_hubert(wav, sr):
    inputs = feature_extractor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = hubert(inputs.input_values, output_hidden_states=True)
        embedding = outputs.hidden_states[9].squeeze(0).mean(dim=0).numpy()
    return embedding

def predict_accent(audio_data, sr):
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=0)
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

    # Define n_fft used in extract_mfcc to determine minimum length for librosa
    n_fft_mfcc = 2048 # Corresponding to n_fft=2048 in extract_mfcc

    # Pad audio if it's too short for MFCC extraction (librosa needs at least n_fft samples)
    if len(audio_data) < n_fft_mfcc:
        pad_length = n_fft_mfcc - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad_length), 'constant')

    # Access max_mfcc_len directly from the global scaler object
    max_mfcc_length = scaler.max_mfcc_len

    mfcc = extract_mfcc(audio_data, 16000, max_len=max_mfcc_length)
    hubert_emb = extract_hubert(audio_data, 16000)
    combined = np.concatenate([mfcc.flatten(), hubert_emb])
    features_scaled = scaler.transform(combined.reshape(1, -1))

    with torch.no_grad():
        outputs = model(torch.tensor(features_scaled, dtype=torch.float32))
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

        # Get all probabilities
        all_probs = {label_encoder.classes_[i]: probs[0, i].item() for i in range(len(label_encoder.classes_))}

    predicted_label = label_encoder.inverse_transform([pred_idx])[0]
    return predicted_label, confidence, all_probs

# ============= INPUT OPTIONS =============
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🎙️ Choose Your Input Method")
option = st.radio("", ["📁 Upload Audio File", "🎤 Record Live Audio"], horizontal=True, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

audio_data, sr = None, None

if option == "📁 Upload Audio File":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Drop your audio file here (WAV, MP3, M4A)", type=['wav', 'mp3', 'm4a'])
    if uploaded_file:
        st.audio(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(uploaded_file.read())
            audio_data, sr = librosa.load(tmp.name, sr=None)
            os.unlink(tmp.name)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**🎤 Click the microphone to start/stop recording:**")
    audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=16000, text="")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    st.markdown('</div>', unsafe_allow_html=True)

# ============= PREDICTION & RESULTS =============
if audio_data is not None:
    with st.spinner("🔍 Analyzing your accent with Deep Learning..."):
        try:
            # Updated call: max_mfcc_len is now accessed inside predict_accent
            pred_label, confidence, all_probs = predict_accent(audio_data, sr)
            info = region_cuisine_map.get(pred_label, {
                'region': 'India 🇮🇳',
                'state': 'General',
                'cuisines': ['Thali', 'Platter'],
                'icon': '🇮🇳',
                'description': 'Diverse Indian cuisine'
            })

            # Result Box
            st.markdown(f"""
            <div class='result-box'>
                <div style='font-size:5rem;'>{info['icon']}</div>
                <h2>{info['region']}</h2>
                <p style='font-size:1.4rem; margin: 1rem 0;'>
                    <strong>Detected Language:</strong> {pred_label}<br>
                    <strong>State:</strong> {info['state']}
                </p>
                <div class='confidence-bar-container'>
                    <div class='confidence-bar' style='width:{confidence*100}%;'></div>
                </div>
                <p style='font-size:1.3rem; margin-top: 1rem;'>
                    <strong>Confidence: {confidence*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Cuisine Recommendations
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🍽️ Personalized Cuisine Recommendations")
            st.markdown(f"*{info['description']}*")
            st.markdown('</div>', unsafe_allow_html=True)

            cols = st.columns(len(info['cuisines']))
            for col, dish in zip(cols, info['cuisines']):
                col.markdown(f"<div class='cuisine-card'><strong>{dish}</strong></div>", unsafe_allow_html=True)

            # Detailed Probabilities
            with st.expander("📊 View Detailed Prediction Probabilities"):
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                for lang, prob in sorted_probs:
                    st.progress(prob, text=f"{lang}: {prob*100:.2f}%")

        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")

# ============= INFO SECTION =============
st.markdown("""
<div class="info-section">
    <h3>🔬 About This System</h3>
    <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem; margin-top: 1.5rem;">
        <div class="stat-box">
            <h4>95+%</h4>
            <p>Accuracy on Adults</p>
        </div>
        <div class="stat-box">
            <h4>6 Languages</h4>
            <p>Indian Languages</p>
        </div>
        <div class="stat-box">
            <h4>HuBERT + MFCC</h4>
            <p>Deep Learning</p>
        </div>
        <div class="stat-box">
            <h4>Cross-Age</h4>
            <p>Generalization</p>
        </div>
    </div>
    <p style="margin-top: 2rem; text-align: center; opacity: 0.9;">
        This system uses state-of-the-art self-supervised learning (HuBERT) combined with traditional
        acoustic features (MFCC) to identify native language from English speech patterns.
    </p>
</div>
""", unsafe_allow_html=True)

# ============= TEAM SECTION =============
st.markdown("""
<div class="team-section">
    <h3>👨‍💻 Research Team</h3>
    <div style="margin-top: 1.5rem;">
        <div class="team-member">🎓 Pangoth Hemanth Nayak</div>
        <div class="team-member">🎓 Arutla Prasanna</div>
        <div class="team-member">🎓 Apurba Nandi</div>
    </div>
    <p style="margin-top: 2rem; opacity: 0.9;">
        🏛️ <strong>IIIT Hyderabad</strong> | NLP Final Project 2025
    </p>
    <p style="opacity: 0.8; margin-top: 1rem;">
        Powered by: PyTorch • HuBERT • Librosa • Streamlit • Neural Networks
    </p>
</div>
""", unsafe_allow_html=True)
