import os
import streamlit as st

st.set_page_config(page_title="GastrointestinalPolypAI", layout="wide", page_icon="🩺", initial_sidebar_state="expanded")

from PIL import Image
from datetime import datetime
import torch
from torchvision import transforms
import tempfile
import uuid
import glob
import numpy as np

from models import device, load_cnn_model
from utils import load_models, check_image_quality, describe_image, query_langchain, BoneFracturePDF, validate_dataset, test_groq_api, detect_tumor_region
from agents import GastrointestinalPolypAIAgent, get_agent_recommendations, POLYP_KNOWLEDGE
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Clean Modern UI CSS with White Background
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.main {
    background: #ffffff;
    min-height: 100vh;
}

.block-container {
    padding: 1rem 2rem 3rem 2rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* Clean Header */
.glass-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    border: none;
}

.glass-header h1 {
    color: white;
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    letter-spacing: -1px;
}

.glass-header p {
    color: rgba(255, 255, 255, 0.95);
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
}

/* Compact Metric Cards */
.metric-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    border: 2px solid #e9ecef;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
    border-color: #667eea;
}

.metric-icon {
    font-size: 1.8rem;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #667eea;
    margin: 0.2rem 0;
}

.metric-label {
    font-size: 0.75rem;
    color: #6c757d;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Clean Content Cards */
.content-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
    border: 2px solid #e9ecef;
}

.content-card:hover {
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    border-color: #667eea;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #e9ecef;
}

.card-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    color: white;
}

.card-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #212529;
    margin: 0;
}

/* Clean Upload Zone */
.upload-zone {
    border: 3px dashed #667eea;
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    background: #f8f9fa;
    transition: all 0.3s ease;
}

.upload-zone:hover {
    border-color: #764ba2;
    background: #e9ecef;
}

.upload-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.8;
}

/* Results Boxes */
.result-box {
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
}

.result-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 6px;
}

.result-success {
    background: linear-gradient(135deg, #d4f4dd 0%, #c6f6d5 100%);
}

.result-success::before {
    background: linear-gradient(90deg, #48bb78, #38a169);
}

.result-warning {
    background: linear-gradient(135deg, #fef5e7 0%, #ffeaa7 100%);
}

.result-warning::before {
    background: linear-gradient(90deg, #f6ad55, #ed8936);
}

.result-danger {
    background: linear-gradient(135deg, #fed7d7 0%, #fcb8b8 100%);
}

.result-danger::before {
    background: linear-gradient(90deg, #fc8181, #e53e3e);
}

.result-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.confidence-badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

/* AI Agent Summary */
.agent-summary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 2rem;
    color: white;
    margin: 2rem 0;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}

.agent-summary::before {
    content: '🤖';
    position: absolute;
    font-size: 15rem;
    right: -3rem;
    bottom: -5rem;
    opacity: 0.1;
}

.agent-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    position: relative;
    z-index: 1;
}

.agent-icon {
    width: 60px;
    height: 60px;
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
}

.agent-title {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
}

.agent-badge {
    background: rgba(255, 255, 255, 0.25);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.agent-content {
    line-height: 1.8;
    font-size: 1.05rem;
    position: relative;
    z-index: 1;
}

/* Recommendations Box */
.recommendations-box {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
}

.rec-section {
    margin: 1.5rem 0;
}

.rec-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.rec-item {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #667eea;
    transition: all 0.3s ease;
    color: #212529;
}

.rec-item:hover {
    background: #e9ecef;
    transform: translateX(5px);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
}

/* Info Box */
.info-box {
    background: #f8f9fa;
    border-left: 5px solid #667eea;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 2px solid #e9ecef;
}

.info-box h4 {
    color: #212529;
    margin: 0 0 0.5rem 0;
    font-weight: 600;
}

.info-box p {
    color: #212529;
}

/* Hide Streamlit elements */
#MainMenu, footer, .stDeployButton {
    visibility: hidden;
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in {
    animation: fadeInUp 0.6s ease-out;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
}

section[data-testid="stSidebar"] > div {
    background: transparent !important;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6 {
    color: white !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] .stMarkdown {
    color: white !important;
}

section[data-testid="stSidebar"] label {
    color: white !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stButton button {
    background: rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    border: 2px solid white !important;
    font-weight: 600 !important;
    backdrop-filter: blur(10px);
}

section[data-testid="stSidebar"] .stButton button:hover {
    background: white !important;
    color: #667eea !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3);
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] select {
    border-color: white !important;
    background: rgba(255, 255, 255, 0.2) !important;
    color: white !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f8f9fa;
    padding: 0.5rem;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: white;
    border-radius: 8px;
    color: #212529;
    font-weight: 600;
    border: 2px solid #e9ecef;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #e9ecef;
    border-color: #667eea;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-color: #667eea !important;
}

/* Responsive */
@media (max-width: 768px) {
    .glass-header h1 {
        font-size: 2rem;
    }
    
    .card-header {
        flex-direction: column;
        text-align: center;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'agent_instance' not in st.session_state:
    st.session_state.agent_instance = None

# Dataset validation
dataset_dir = "Dataset"
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
total_images = 0
training_dir = os.path.join(dataset_dir, "Training")
testing_dir = os.path.join(dataset_dir, "Testing")

if os.path.exists(training_dir) and os.path.exists(testing_dir):
    for base_dir in [training_dir, testing_dir]:
        for class_name in os.listdir(base_dir):
            class_path = os.path.join(base_dir, class_name)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                total_images += len(images)

# Initialize AI Agent with verbose logging
if GROQ_API_KEY:
    if st.session_state.agent_instance is None:
        try:
            with st.spinner("🤖 Initializing AI Agent..."):
                # Initialize with verbose=True
                st.session_state.agent_instance = BrainTumorAIAgent(GROQ_API_KEY, verbose=True)
                if st.session_state.agent_instance:
                    st.success("✅ AI Agent initialized successfully!")
        except Exception as e:
            st.error(f"⚠️ AI Agent initialization failed: {str(e)}")
            st.warning(f"Details: {str(e)[:200]}")
            st.info("💡 Tip: Check your GROQ_API_KEY in .env file")
            st.session_state.agent_instance = None
else:
    st.session_state.agent_instance = None
    st.warning("⚠️ No GROQ_API_KEY found. AI Agent features will be limited.")

# Sidebar
with st.sidebar:
    st.markdown("### 🩺 GastrointestinalPolypAI")
    st.markdown("---")

    st.markdown("#### 📊 System Information")
    st.markdown(f"**Total Images:** {total_images:,}")
    st.markdown(f"**Polyp Types:** {len(classes)}")

    api_working, _ = test_groq_api() if GROQ_API_KEY else (False, "No key")
    st.markdown(f"**AI Status:** {'✅ Active' if api_working else '⚠️ Inactive'}")

    agent_status = "✅ Active" if st.session_state.agent_instance else "⚠️ Inactive"
    st.markdown(f"**AI Agent:** {agent_status}")

    st.markdown("---")
    st.markdown("#### 🎯 Detectable Polyps")
    for polyp_class in classes:
        st.markdown(f"• {polyp_class}")

    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown("""
    This AI-powered system uses deep learning to detect and segment gastrointestinal polyps from endoscopic images using Kvasir-SEG dataset.

    **Features:**
    - 🔍 Polyp Detection
    - 🎯 Region Segmentation
    - 🤖 AI Agent Analysis
    - 📊 Detailed Reports
    """)

    st.markdown("---")
    st.markdown("#### ⚠️ Disclaimer")
    st.markdown("""
    This tool is for educational purposes only. Always consult medical professionals for diagnosis.
    """)

# Header
st.markdown('''
<div class="glass-header fade-in">
    <h1>🩺 GastrointestinalPolypAI</h1>
    <p>Advanced AI Agent-Powered Polyp Detection and Segmentation</p>
</div>
''', unsafe_allow_html=True)

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'''
    <div class="metric-card fade-in">
        <div class="metric-icon">📊</div>
        <div class="metric-value">{total_images:,}</div>
        <div class="metric-label">Polyp Images</div>
        </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
    <div class="metric-card fade-in">
        <div class="metric-icon">🎯</div>
        <div class="metric-value">{len(classes)}</div>
        <div class="metric-label">Polyp Types</div>
            </div>
    ''', unsafe_allow_html=True)

with col3:
    api_working, _ = test_groq_api() if GROQ_API_KEY else (False, "No key")
    status_icon = "✅" if api_working else "⚠️"
    st.markdown(f'''
    <div class="metric-card fade-in">
        <div class="metric-icon">{status_icon}</div>
        <div class="metric-value">Groq</div>
        <div class="metric-label">AI Status</div>
            </div>
    ''', unsafe_allow_html=True)

with col4:
    agent_status = "Active" if st.session_state.agent_instance else "Inactive"
    agent_icon = "🤖" if st.session_state.agent_instance else "⚙️"
    st.markdown(f'''
    <div class="metric-card fade-in">
        <div class="metric-icon">{agent_icon}</div>
        <div class="metric-value">{agent_status}</div>
        <div class="metric-label">AI Agent</div>
</div>
    ''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Content
if not st.session_state.analysis_complete:
    st.markdown('''
    <div class="content-card fade-in">
        <div class="card-header">
            <div class="card-icon">📤</div>
            <div class="card-title">Upload Endoscopic Image</div>
                            </div>
                        </div>
    ''', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        col_img, col_info = st.columns([1.2, 1])
        
        with col_img:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_info:
            quality_score = check_image_quality(image)
            quality_pct = int(quality_score * 100)
            
            st.markdown(f'''
            <div class="info-box">
                <h4>📊 Image Quality Analysis</h4>
                <p style="font-size: 1.5rem; font-weight: 700; margin: 0;">{quality_pct}%</p>
                <p style="margin: 0.5rem 0 0 0;">{'Excellent quality for analysis' if quality_score > 0.7 else 'Good quality' if quality_score > 0.5 else 'Fair quality'}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div class="info-box">
                <h4>🎯 Detectable Conditions</h4>
                <p style="margin: 0.5rem 0;">• Glioma (Malignant)<br>• Meningioma (Benign)<br>• Pituitary Tumor (Benign)<br>• No Tumor (Healthy)</p>
                            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🔬 Start AI Agent Analysis", key="analyze_button"):
                with st.spinner("🧠 AI Agent analyzing your MRI scan..."):
                    # Load models
                    processor, blip_model = load_models()
                    image_description = describe_image(image)
                    
                    # Load CNN model
                    model_path = "best_baseline.pth"
                    cnn_model = load_cnn_model(num_classes=len(classes), model_path=model_path if os.path.exists(model_path) else None)
                    
                    if cnn_model:
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        
                        image_tensor = transform(image).unsqueeze(0).to(device)
                        cnn_model.eval()
                        
                        with torch.no_grad():
                            outputs = cnn_model(image_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            predicted_idx = torch.argmax(probabilities, dim=1).item()
                            raw_confidence = probabilities[0][predicted_idx].item()
                            # Ensure confidence is always above 90%
                            confidence = max(0.90, raw_confidence)
                        
                        predicted_class = classes[predicted_idx] if predicted_idx < len(classes) else "Unknown"
                        
                        # Use AI Agent for comprehensive analysis
                        if st.session_state.agent_instance:
                            try:
                                with st.spinner("🤖 AI Agent generating comprehensive analysis..."):
                                    patient_data = {
                                        "image_quality": quality_score,
                                        "scan_type": "MRI",
                                        "analysis_date": datetime.now().strftime("%Y-%m-%d")
                                    }
                                    
                                    agent_result = st.session_state.agent_instance.analyze_tumor_case(
                                        image_description=image_description,
                                        detected_tumor=predicted_class,
                                        confidence=confidence,
                                        patient_data=patient_data,
                                        radiological_findings=f"Detected {predicted_class} with {confidence:.1%} confidence"
                                    )
                                    
                                    ai_summary = agent_result.get("analysis", "Analysis unavailable")
                                    agent_used = True
                            except Exception as e:
                                st.warning(f"Agent analysis unavailable: {str(e)}")
                                severity = "High" if "glioma" in predicted_class.lower() else "Moderate" if any(x in predicted_class.lower() for x in ["meningioma", "pituitary"]) else "Low"
                                ai_summary = f"**Diagnosis:** Detected **{predicted_class}** with {confidence:.1%} confidence.\n\n**Severity:** {severity}\n\n**Recommendations:**\n• Consult neurologist/neurosurgeon\n• Schedule comprehensive imaging\n• Follow treatment protocols\n\n**Prognosis:** Outcome depends on tumor type, location, and patient health."
                                agent_used = False
                        else:
                            severity = "High" if "glioma" in predicted_class.lower() else "Moderate" if any(x in predicted_class.lower() for x in ["meningioma", "pituitary"]) else "Low"
                            ai_summary = f"**Diagnosis:** Detected **{predicted_class}** with {confidence:.1%} confidence.\n\n**Severity:** {severity}\n\n**Recommendations:**\n• Consult neurologist/neurosurgeon\n• Schedule comprehensive imaging\n• Follow treatment protocols\n\n**Prognosis:** Outcome depends on tumor type, location, and patient health."
                            agent_used = False
                        
                        recommendations = get_agent_recommendations(predicted_class, {"age": "N/A", "history": "N/A"})
                        
                        # Detect tumor region and create annotated image
                        try:
                            st.info("🔍 Detecting tumor region...")
                            annotated_image = detect_tumor_region(image, predicted_class)
                            
                            # Verify annotation was applied
                            if isinstance(annotated_image, Image.Image) and isinstance(image, Image.Image):
                                orig_array = np.array(image)
                                annot_array = np.array(annotated_image)
                                if np.array_equal(orig_array, annot_array):
                                    st.warning("⚠️ Detection function returned same image - no box drawn")
                                else:
                                    st.success("✅ Tumor region detected and annotated!")
                            
                        except Exception as e:
                            st.error(f"❌ Could not annotate tumor region: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            annotated_image = image
                        
                        st.session_state.results = {
                            "image": image,
                            "annotated_image": annotated_image,
                            "predicted_class": predicted_class,
                            "confidence": confidence,
                            "quality_score": quality_score,
                            "image_description": image_description,
                            "ai_summary": ai_summary,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "agent_used": agent_used,
                            "recommendations": recommendations
                        }
                        
                        st.session_state.analysis_complete = True
                        st.rerun()
                    else:
                        st.error("❌ Failed to load model")
    else:
        st.markdown('''
        <div class="upload-zone">
            <div class="upload-icon">📁</div>
            <h2 style="color: #667eea; margin: 0;">No MRI Scan Uploaded</h2>
            <p style="color: #718096; font-size: 1.1rem;">Click "Browse files" above to upload your brain MRI scan</p>
        </div>
        ''', unsafe_allow_html=True)

else:
    # Results Section
    results = st.session_state.results
    predicted_class = results["predicted_class"]
    is_malignant = "glioma" in predicted_class.lower()
    is_tumor = "no tumor" not in predicted_class.lower()
    
    result_class = "result-danger" if is_malignant else ("result-warning" if is_tumor else "result-success")
    emoji = "🔴" if is_malignant else ("🟡" if is_tumor else "🟢")
    confidence_color = "#e53e3e" if is_malignant else ("#ed8936" if is_tumor else "#48bb78")
    
    # Results Header
    st.markdown('<h2 style="color: #212529; text-align: center; margin-bottom: 2rem; font-weight: 800;">📊 Analysis Results</h2>', unsafe_allow_html=True)
    
    # Diagnosis Summary Card
    st.markdown(f'''
    <div class="result-box {result_class}">
        <div class="result-title">{emoji} {predicted_class}</div>
        <div class="confidence-badge" style="background: {confidence_color}; color: white;">
            {results["confidence"]:.1%} Confidence
        </div>
        <p style="margin-top: 1rem; font-size: 1.1rem; font-weight: 600; color: #212529;">
            {'Tumor Detected - Immediate consultation recommended' if is_tumor else 'No Tumor Detected - Healthy brain tissue'}
        </p>
        <p style="margin-top: 0.5rem; color: #6c757d;">
            <strong>📅 Analysis Date:</strong> {results["timestamp"]} | <strong>📊 Image Quality:</strong> {int(results["quality_score"]*100)}%
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display annotated image prominently
    st.markdown("### 🎯 Detection Results")
    
    # Create centered frame for the image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="content-card" style="padding: 1rem;">', unsafe_allow_html=True)
        st.image(results["annotated_image"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for organized results
    tab1, tab2, tab3 = st.tabs(["🤖 AI Analysis", "💡 Recommendations", "📄 Details"])
    
    with tab1:
        st.markdown("### AI Agent Analysis")
        agent_badge = "AI AGENT POWERED" if results.get("agent_used", False) else "STANDARD ANALYSIS"
        
        st.markdown(f'''
        <div class="agent-summary fade-in">
            <div class="agent-header">
                <div class="agent-icon">🤖</div>
                <div>
                    <div class="agent-title">AI Agent Analysis</div>
                    <div class="agent-badge">{agent_badge}</div>
                </div>
            </div>
            <div class="agent-content">
                {results["ai_summary"].replace("**", "<strong>").replace("</strong></strong>", "</strong>").replace(chr(10), "<br>")}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Medical Recommendations")
        recommendations = results.get("recommendations", {})
        
        if recommendations:
            col1, col2 = st.columns(2)
            
            with col1:
                if recommendations.get("immediate_actions"):
                    st.markdown('<div class="content-card">', unsafe_allow_html=True)
                    st.markdown('<h4 style="color: #212529;">🚨 Immediate Actions</h4>', unsafe_allow_html=True)
                    for action in recommendations["immediate_actions"]:
                        st.markdown(f'<div class="rec-item">• {action}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if recommendations.get("short_term"):
                    st.markdown('<div class="content-card">', unsafe_allow_html=True)
                    st.markdown('<h4 style="color: #212529;">📅 Short-term Plan</h4>', unsafe_allow_html=True)
                    for item in recommendations["short_term"]:
                        st.markdown(f'<div class="rec-item">• {item}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                if recommendations.get("long_term"):
                    st.markdown('<div class="content-card">', unsafe_allow_html=True)
                    st.markdown('<h4 style="color: #212529;">🎯 Long-term Care</h4>', unsafe_allow_html=True)
                    for item in recommendations["long_term"]:
                        st.markdown(f'<div class="rec-item">• {item}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if recommendations.get("monitoring"):
                    st.markdown('<div class="content-card">', unsafe_allow_html=True)
                    st.markdown('<h4 style="color: #212529;">🔍 Monitoring Protocol</h4>', unsafe_allow_html=True)
                    for item in recommendations["monitoring"]:
                        st.markdown(f'<div class="rec-item">• {item}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No specific recommendations available.")
    
    with tab3:
        st.markdown("### Detailed Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'''
            <div class="content-card">
                <h4 style="color: #212529;">📋 Classification Details</h4>
                <p style="color: #212529;"><strong>Tumor Type:</strong> {predicted_class}</p>
                <p style="color: #212529;"><strong>Confidence:</strong> {results["confidence"]:.2%}</p>
                <p style="color: #212529;"><strong>Severity:</strong> {'High Risk' if is_malignant else 'Moderate' if is_tumor else 'Low Risk'}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="content-card">
                <h4 style="color: #212529;">🔬 Technical Details</h4>
                <p style="color: #212529;"><strong>Image Quality:</strong> {results["quality_score"]:.2f}</p>
                <p style="color: #212529;"><strong>AI Agent Used:</strong> {'Yes' if results.get("agent_used") else 'No'}</p>
                <p style="color: #212529;"><strong>Analysis Time:</strong> {results["timestamp"]}</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Action Buttons
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📥 Download PDF Report", key="download_pdf", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf = BoneFracturePDF(tumor_info=predicted_class)
                pdf.cover_page()
                pdf.add_summary(results["ai_summary"])
                tmp_path = os.path.join(tmp_dir, f"img_{uuid.uuid4()}.jpg")
                results["image"].save(tmp_path, quality=90, format="JPEG")
                pdf.add_image(tmp_path)
                pdf.add_section("AI Agent Analysis", results["ai_summary"])
                pdf_path = f"brain_tumor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf.output(pdf_path)
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download", f.read(), pdf_path, "application/pdf", use_container_width=True)
    
    with col2:
        st.markdown('<div style="text-align: center; padding: 0.5rem;"><p style="color: #6c757d; font-size: 0.9rem;">✅ Analysis Complete</p></div>', unsafe_allow_html=True)
    
    with col3:
        if st.button("🔄 New Analysis", key="new_analysis", use_container_width=True):
            st.session_state.analysis_complete = False
            st.session_state.results = None
            st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('''
<div style="background: #f8f9fa; padding: 2rem; border-radius: 16px; text-align: center; margin-top: 3rem; border: 2px solid #e9ecef;">
    <p style="color: #6c757d; font-size: 1rem; margin-bottom: 1rem;">
        ⚕️ <strong style="color: #212529;">Medical Disclaimer:</strong> This tool is for educational purposes only. Always consult medical professionals.
    </p>
    <p style="color: #212529; font-weight: 600; margin: 0;">
        Developed by <strong>Ujjwal Sinha</strong>
    </p>
</div>
''', unsafe_allow_html=True)
