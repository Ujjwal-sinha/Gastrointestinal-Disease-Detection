"""
AI Agents for BrainTumorAI - Brain Tumor Analysis Platform
Enhanced with intelligent agents for comprehensive brain tumor analysis
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st
import time
import random

# LangChain imports
from langchain.agents import initialize_agent
from langchain.tools import BaseTool
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

def retry_with_exponential_backoff(func, max_retries=3, base_delay=1):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
    
    Returns:
        Result of the function call
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_msg = str(e).lower()
            
            # If it's not a capacity issue, don't retry
            if "over capacity" not in error_msg and "503" not in str(e):
                raise e
            
            if attempt == max_retries:
                raise e
            
            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"GROQ API over capacity, retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries + 1})")
            time.sleep(delay)

# Brain tumor knowledge base
BRAIN_TUMOR_KNOWLEDGE = {
    "notumor": {
        "symptoms": ["normal brain structure", "no abnormal masses", "clear brain tissue"],
        "characteristics": ["healthy brain tissue", "normal ventricles", "no lesions"],
        "recommendations": ["maintain brain health", "regular checkups", "healthy lifestyle"]
    },
    "glioma": {
        "symptoms": ["headaches", "seizures", "cognitive changes", "vision problems", "weakness"],
        "characteristics": ["originates from glial cells", "can be low or high grade", "infiltrative growth"],
        "causes": ["genetic mutations", "exposure to radiation", "family history"],
        "treatments": ["surgery", "radiation therapy", "chemotherapy", "targeted therapy"],
        "severity": "high",
        "type": "malignant",
        "description": "Gliomas are tumors that originate from glial cells in the brain. They can be benign or malignant."
    },
    "meningioma": {
        "symptoms": ["headaches", "vision changes", "hearing loss", "memory loss", "seizures"],
        "characteristics": ["arises from meninges", "usually benign", "slow-growing"],
        "causes": ["radiation exposure", "genetic factors", "hormone levels"],
        "treatments": ["observation", "surgery", "radiation therapy", "stereotactic radiosurgery"],
        "severity": "moderate",
        "type": "benign (usually)",
        "description": "Meningiomas are tumors that arise from the meninges, the membranes surrounding the brain and spinal cord."
    },
    "pituitary": {
        "symptoms": ["hormonal imbalances", "vision problems", "headaches", "fatigue", "irregular menstruation"],
        "characteristics": ["located in pituitary gland", "affects hormone production", "usually benign"],
        "causes": ["genetic mutations", "hormonal factors", "unknown causes"],
        "treatments": ["medication", "surgery", "radiation therapy", "hormone replacement"],
        "severity": "moderate",
        "type": "benign (usually)",
        "description": "Pituitary tumors develop in the pituitary gland and can affect hormone production and regulation."
    }
}

class BrainImageAnalysisTool(BaseTool):
    name: str = "brain_image_analyzer"
    description: str = """Analyze MRI images for brain tumor detection and assessment with enhanced accuracy. 
    Input should be a JSON string with keys: 'image_description', 'detected_tumor', and 'confidence'."""
    
    def _run(self, query: str) -> str:
        """Analyze MRI image and provide detailed insights with enhanced tumor detection"""
        
        # Parse input JSON
        try:
            import json
            data = json.loads(query)
            image_description = data.get("image_description", "")
            detected_tumor = data.get("detected_tumor", "")
            confidence = float(data.get("confidence", 0.0))
        except:
            return "Error: Invalid input format. Please provide JSON with 'image_description', 'detected_tumor', and 'confidence'."
        
        # Enhanced tumor analysis with multiple factors
        tumor_analysis = self._enhanced_tumor_analysis(detected_tumor, confidence, image_description)
        
        analysis = {
            "tumor": detected_tumor,
            "confidence": confidence,
            "enhanced_confidence": tumor_analysis["enhanced_confidence"],
            "severity": tumor_analysis["severity"],
            "tumor_type": tumor_analysis["tumor_type"],
            "location": tumor_analysis["location"],
            "treatment_timeline": tumor_analysis["treatment_timeline"],
            "symptoms": self._get_symptoms(detected_tumor),
            "recommendations": self._get_recommendations(detected_tumor),
            "risk_level": self._assess_risk(detected_tumor, confidence),
            "treatment_urgency": tumor_analysis["treatment_urgency"],
            "complications_risk": tumor_analysis["complications_risk"]
        }
        return json.dumps(analysis, indent=2)
    
    async def _arun(self, query: str) -> str:
        """Async version of _run"""
        return self._run(query)
    
    def _enhanced_tumor_analysis(self, tumor: str, confidence: float, image_description: str) -> dict:
        """Enhanced tumor analysis with multiple assessment factors"""
        tumor_lower = tumor.lower().replace(' ', '_')
        
        # Base analysis from knowledge base
        base_info = BRAIN_TUMOR_KNOWLEDGE.get(tumor_lower, {})
        
        # Enhanced confidence calculation
        enhanced_confidence = confidence
        if "mass" in image_description.lower() or "lesion" in image_description.lower():
            if any(tumor_type in tumor_lower for tumor_type in ["glioma", "meningioma", "pituitary"]):
                enhanced_confidence = min(0.98, confidence * 1.15)
        
        if "abnormal" in image_description.lower() or "irregular" in image_description.lower():
            if any(ttype in tumor_lower for ttype in ["glioma", "meningioma"]):
                enhanced_confidence = min(0.97, confidence * 1.10)
        
        # Determine tumor characteristics
        tumor_type = base_info.get("type", "Unknown")
        
        # Estimate location based on tumor type
        location = "Brain tissue"
        if "glioma" in tumor_lower:
            location = "Glial cells (cerebral hemispheres)"
        elif "meningioma" in tumor_lower:
            location = "Meninges (brain covering)"
        elif "pituitary" in tumor_lower:
            location = "Pituitary gland (brain base)"
        
        # Treatment timeline estimation
        treatment_timeline = self._estimate_treatment_time(tumor_lower, tumor_type)
        
        # Treatment urgency
        treatment_urgency = "Routine"
        if "glioma" in tumor_lower:
            treatment_urgency = "Urgent"
        elif "meningioma" in tumor_lower or "pituitary" in tumor_lower:
            treatment_urgency = "Semi-urgent"
        
        # Complications risk
        complications_risk = "Low"
        if "glioma" in tumor_lower:
            complications_risk = "High"
        elif "meningioma" in tumor_lower:
            complications_risk = "Moderate"
        elif "pituitary" in tumor_lower:
            complications_risk = "Moderate"
        
        return {
            "enhanced_confidence": enhanced_confidence,
            "severity": base_info.get("severity", "moderate"),
            "tumor_type": tumor_type,
            "location": location,
            "treatment_timeline": treatment_timeline,
            "treatment_urgency": treatment_urgency,
            "complications_risk": complications_risk
        }
    
    def _estimate_treatment_time(self, tumor_type: str, complexity: str) -> str:
        """Estimate treatment timeline based on tumor characteristics"""
        if "notumor" in tumor_type:
            return "No treatment needed (healthy brain)"
        elif "glioma" in tumor_type:
            return "12-18 months (surgery + chemo/radiation)"
        elif "meningioma" in tumor_type:
            return "3-6 months (surgery or observation)"
        elif "pituitary" in tumor_type:
            return "6-12 months (medication or surgery)"
        else:
            return "Varies based on tumor type and grade"
    
    def _assess_severity(self, confidence: float, tumor: str) -> str:
        tumor_lower = tumor.lower().replace(' ', '_')
        if tumor_lower in BRAIN_TUMOR_KNOWLEDGE:
            return BRAIN_TUMOR_KNOWLEDGE[tumor_lower].get("severity", "unknown")
        elif confidence > 0.9:
            return "High"
        elif confidence > 0.7:
            return "Moderate"
        else:
            return "Low"
    
    def _get_symptoms(self, tumor: str) -> List[str]:
        tumor_lower = tumor.lower().replace(' ', '_')
        if tumor_lower in BRAIN_TUMOR_KNOWLEDGE:
            return BRAIN_TUMOR_KNOWLEDGE[tumor_lower].get("symptoms", [])
        return ["Consult neurologist or neurosurgeon for specific symptoms"]
    
    def _get_recommendations(self, tumor: str) -> List[str]:
        tumor_lower = tumor.lower().replace(' ', '_')
        if tumor_lower in BRAIN_TUMOR_KNOWLEDGE:
            return BRAIN_TUMOR_KNOWLEDGE[tumor_lower].get("treatments", [])
        return ["Seek professional neurological consultation"]
    
    def _assess_risk(self, tumor: str, confidence: float) -> str:
        """Assess risk level based on tumor type"""
        tumor_lower = tumor.lower().replace(' ', '_')
        if "glioma" in tumor_lower:
            return "High"
        elif "meningioma" in tumor_lower or "pituitary" in tumor_lower:
            return "Moderate"
        elif "notumor" in tumor_lower:
            return "Low"
        else:
            return "Unknown"

class BrainTumorAIAgent:
    """Main Brain Tumor AI Agent with Enhanced Detection"""
    
    def __init__(self, api_key: str, verbose: bool = True):
        print(f"🤖 Initializing BrainTumorAIAgent with verbose={verbose}...")
        self.api_key = api_key
        self.verbose = verbose
        
        try:
            print("📡 Getting working LLM...")
            self.llm = self._get_working_llm()
            print(f"✅ LLM initialized: {type(self.llm).__name__}")
        except Exception as e:
            print(f"❌ Failed to initialize LLM: {str(e)}")
            raise Exception(f"LLM initialization failed: {str(e)}")
        
        try:
            print("🧠 Creating memory buffer...")
            self.memory = ConversationBufferMemory(memory_key="chat_history")
            print("✅ Memory buffer created")
        except Exception as e:
            print(f"❌ Failed to create memory: {str(e)}")
            raise Exception(f"Memory creation failed: {str(e)}")
        
        # Initialize enhanced tools
        try:
            print("🔧 Initializing tools...")
            self.tools = [
                BrainImageAnalysisTool()
            ]
            print(f"✅ Tools initialized: {len(self.tools)} tool(s)")
        except Exception as e:
            print(f"❌ Failed to initialize tools: {str(e)}")
            raise Exception(f"Tools initialization failed: {str(e)}")
        
        # Initialize agent with enhanced capabilities
        try:
            print("🚀 Initializing agent...")
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent="conversational-react-description",
                memory=self.memory,
                verbose=self.verbose,
                handle_parsing_errors=True
            )
            print("✅ Agent initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {str(e)}")
            raise Exception(f"Agent initialization failed: {str(e)}")
    
    def _get_working_llm(self):
        """Get a working LLM instance with fallback models and retry logic."""
        models_to_try = [
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        print(f"🔍 Testing {len(models_to_try)} Groq models...")
        
        for i, model_name in enumerate(models_to_try, 1):
            try:
                print(f"  [{i}/{len(models_to_try)}] Trying {model_name}...")
                
                def test_model():
                    llm = ChatGroq(
                        model=model_name,
                        temperature=0.1,
                        groq_api_key=self.api_key
                    )
                    # Test the model with a simple prompt
                    test_response = llm.invoke("Test")
                    if test_response:
                        print(f"  ✅ {model_name} is working!")
                        return llm
                    else:
                        raise Exception("Empty response from model")
                
                # Use retry mechanism for this model
                return retry_with_exponential_backoff(test_model)
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"  ⚠️ {model_name} failed: {str(e)[:100]}")
                if "over capacity" in error_msg or "503" in str(e):
                    print(f"  ⏳ Model over capacity, trying next...")
                    continue
                else:
                    # For other errors, try next model
                    print(f"  ⏭️ Trying next model...")
                    continue
        
        # If all models fail, raise an exception
        error_msg = "All GROQ models are currently unavailable. Please check your API key or try again later."
        print(f"❌ {error_msg}")
        raise Exception(error_msg)
    
    def analyze_tumor_case(self, 
                             image_description: str,
                             detected_tumor: str,
                             confidence: float,
                             patient_data: Dict,
                             radiological_findings: str = "",
                             detection_metadata: Dict = None) -> Dict:
        """Comprehensive brain tumor case analysis with enhanced accuracy"""
        
        print("=" * 60)
        print("🔬 Starting AI Agent tumor case analysis...")
        print(f"📋 Detected Tumor: {detected_tumor}")
        print(f"📊 Confidence: {confidence:.2%}")
        print("=" * 60)
        
        # Enhanced analysis prompt with multiple assessment factors
        prompt = f"""
        Analyze this brain tumor case comprehensively with enhanced accuracy:
        
        Image Description: {image_description}
        Detected Tumor: {detected_tumor}
        Detection Confidence: {confidence}
        Patient Data: {patient_data}
        Radiological Findings: {radiological_findings}
        Detection Metadata: {detection_metadata}
        
        Provide a detailed multi-factor analysis including:
        1. Tumor classification and severity assessment
        2. Radiological finding correlation and validation
        3. Enhanced treatment recommendations with surgical considerations
        4. Comprehensive risk assessment and complication prediction
        5. Treatment timeline and monitoring protocols
        6. Follow-up strategies and imaging recommendations
        7. Patient-specific considerations and modifications
        8. Quality assurance and confidence validation
        
        Use enhanced tumor detection algorithms and cross-reference multiple diagnostic criteria.
        """
        
        try:
            print("🤖 Running agent with prompt...")
            response = self.agent.run(prompt)
            print("✅ Agent analysis completed successfully!")
            print("=" * 60)
            
            return {
                "analysis": response,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "agent_version": "2.0_enhanced"
            }
        except Exception as e:
            print(f"❌ Agent analysis failed: {str(e)}")
            print("🔄 Using fallback analysis...")
            print("=" * 60)
            return {
                "error": str(e),
                "fallback_analysis": self._generate_fallback_analysis(
                    detected_tumor, confidence, patient_data
                )
            }
    
    def _generate_fallback_analysis(self, tumor: str, confidence: float, patient_data: Dict) -> str:
        """Generate fallback analysis when agent fails"""
        return f"""
        **Fallback Brain Tumor Analysis**
        
        Tumor Type: {tumor}
        Confidence: {confidence:.2f}
        
        **Immediate Recommendations:**
        1. Seek immediate neurological consultation
        2. Schedule comprehensive MRI with contrast
        3. Consult with neurosurgeon for treatment options
        4. Monitor symptoms and neurological status
        5. Consider genetic testing if indicated
        
        **Follow-up Actions:**
        - Obtain additional imaging if needed
        - Monitor for symptom progression
        - Begin appropriate treatment protocol
        - Consider multidisciplinary tumor board review
        
        **Note:** This is a preliminary analysis. Professional neurological evaluation is essential for accurate diagnosis and treatment planning.
        """

# Utility functions
def create_agent_instance(agent_type: str, api_key: str):
    """Create agent instance based on type"""
    if agent_type in ["tumor", "brain", "mri"]:
        return BrainTumorAIAgent(api_key)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def get_agent_recommendations(tumor: str, patient_data: Dict) -> Dict:
    """Get enhanced agent recommendations for a tumor"""
    recommendations = {
        "immediate_actions": [],
        "short_term": [],
        "long_term": [],
        "monitoring": [],
        "enhanced_protocols": []
    }
    
    # Tumor-specific recommendations with enhanced protocols
    tumor_lower = tumor.lower()
    
    if "glioma" in tumor_lower:
        recommendations["immediate_actions"].extend([
            "Seek immediate neurosurgical consultation",
            "Schedule comprehensive MRI with contrast",
            "Begin steroid therapy if indicated"
        ])
        recommendations["short_term"].extend([
            "Surgical planning and intervention",
            "Post-operative monitoring and care",
            "Initiate radiation and chemotherapy protocols"
        ])
        recommendations["long_term"].extend([
            "Ongoing chemotherapy and radiation",
            "Regular MRI surveillance",
            "Neurological function monitoring"
        ])
    
    elif "meningioma" in tumor_lower:
        recommendations["immediate_actions"].extend([
            "Neurosurgical consultation for assessment",
            "MRI surveillance protocol establishment"
        ])
        recommendations["short_term"].extend([
            "Observation or surgical planning",
            "Symptom management",
            "Treatment decision making"
        ])
    
    elif "pituitary" in tumor_lower:
        recommendations["immediate_actions"].extend([
            "Endocrinology and neurosurgery consultation",
            "Hormone level assessment"
        ])
        recommendations["short_term"].extend([
            "Medical management or surgical planning",
            "Hormone replacement therapy if needed"
        ])
    
    # Universal monitoring recommendations
    recommendations["monitoring"].extend([
        "Regular MRI follow-up",
        "Neurological status assessment",
        "Symptom monitoring",
        "Quality of life evaluation"
    ])
    
    # Enhanced protocols for all tumors
    recommendations["enhanced_protocols"].extend([
        "Evidence-based treatment protocols",
        "Patient-specific risk assessment",
        "Multidisciplinary team approach",
        "Quality outcome measures"
    ])
    
    return recommendations
