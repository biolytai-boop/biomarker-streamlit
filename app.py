"""
Biomarker Extraction Streamlit App
Uses Qwen3.5-0.8B with LoRA adapter from HuggingFace
"""

import streamlit as st
import torch
import json
import re
from typing import Optional, Dict, List, Any

# Try to import unsloth, fall back to standard transformers if not available
UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    pass

# Constants
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
ADAPTER_NAME = "Shubh-0789/biomarker-qwen3.5-0.8b-lora-v2"
MAX_SEQ_LENGTH = 2048

# System prompt for biomarker extraction
SYSTEM_PROMPT = """You are a biomedical expert specializing in biomarker extraction. 
Your task is to extract biomarkers, medical entities, and clinical measurements from the given text.
Return the results in structured JSON format with the following fields:
- biomarkers: list of biomarker names found
- values: list of measured values with units
- entities: list of other medical/clinical entities
- conditions: list of diseases or conditions mentioned

Be precise and only extract information that is explicitly stated in the text."""

# Example clinical texts for testing
EXAMPLE_TEXTS = [
    "Patient shows elevated CRP levels at 12.5 mg/L indicating possible inflammation. Blood glucose measured at 126 mg/dL suggesting pre-diabetic condition. LDL cholesterol 145 mg/dL.",
    "Lab results: Hemoglobin A1c 7.2%, serum creatinine 1.1 mg/dL, eGFR 85 mL/min/1.73m2. Troponin I levels normal at 0.02 ng/mL.",
    "Tumor marker CA-125 elevated to 45 U/mL. PSA 3.2 ng/mL within normal range. CEA 2.5 μg/L.",
]


@st.cache_resource
def load_model():
    """Load the Qwen3.5-0.8B model with LoRA adapter."""
    try:
        if UNSLOTH_AVAILABLE:
            # Use Unsloth for efficient loading with bf16 (NOT 4-bit)
            # First load the base model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=False,  # IMPORTANT: No 4-bit for Qwen3.5
                load_in_16bit=True,  # Use bf16 loading
                device_map="auto",
            )
            
            # Load the LoRA adapter using PeftModel.from_pretrained
            # This properly loads the adapter weights on top of the base model
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ADAPTER_NAME)
            
            # Enable inference mode for Unsloth
            FastLanguageModel.for_inference(model)
            
        else:
            # Fallback to standard transformers with bf16
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            # Load base model in bf16 (NOT 4-bit)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            # Load and apply LoRA adapter using PeftModel.from_pretrained
            model = PeftModel.from_pretrained(model, ADAPTER_NAME)
            
            # Set to evaluation mode
            model.eval()
            
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def format_prompt(text: str) -> str:
    """Format the input text as a prompt for biomarker extraction using chat template."""
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Extract biomarkers from this clinical text: {text}<|im_end|>
<|im_start|>assistant
"""


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from model response, handling various formats."""
    # Try direct JSON parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block in the response
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Try to extract brace-delimited content
    brace_start = response.find('{')
    brace_end = response.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_start < brace_end:
        try:
            return json.loads(response[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    
    return None


def run_inference(text: str, model, tokenizer) -> Optional[Dict[str, Any]]:
    """Run inference on the input text."""
    if model is None or tokenizer is None:
        return None
    
    try:
        # Format prompt
        prompt = format_prompt(text)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the assistant's response (after the prompt)
        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker in response:
            response = response.split(assistant_marker)[-1]
        
        # Extract JSON from response
        result = extract_json_from_response(response)
        
        return result
        
    except Exception as e:
        st.error(f"Inference error: {str(e)}")
        return None


def display_results(results: Dict[str, Any]):
    """Display the extracted biomarkers in a nice format."""
    if not results:
        st.warning("No biomarkers could be extracted from the text.")
        return
    
    # Create columns for different entity types
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧬 Biomarkers")
        biomarkers = results.get("biomarkers", [])
        if biomarkers:
            for item in biomarkers:
                st.markdown(f"- **{item}**")
        else:
            st.info("No biomarkers found")
        
        st.subheader("📊 Values")
        values = results.get("values", [])
        if values:
            for item in values:
                st.markdown(f"- {item}")
        else:
            st.info("No values found")
    
    with col2:
        st.subheader("🏥 Medical Entities")
        entities = results.get("entities", [])
        if entities:
            for item in entities:
                st.markdown(f"- **{item}**")
        else:
            st.info("No entities found")
        
        st.subheader("🦠 Conditions")
        conditions = results.get("conditions", [])
        if conditions:
            for item in conditions:
                st.markdown(f"- **{item}**")
        else:
            st.info("No conditions found")


def main():
    st.set_page_config(
        page_title="Biomarker Extraction",
        page_icon="🧬",
        layout="wide",
    )
    
    st.title("🧬 Biomarker Extraction Tool")
    st.markdown("""
    Extract biomarkers, medical entities, and clinical measurements from biomedical text 
    using the Qwen3.5-0.8B model fine-tuned for biomarker extraction.
    """)
    
    # Initialize session state
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    
    # Load model
    with st.spinner("Loading model... This may take a few minutes on first run."):
        model, tokenizer = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check your internet connection and try again.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Input section
    st.divider()
    
    # Example texts dropdown
    with st.expander("📝 Try an example text"):
        for i, example in enumerate(EXAMPLE_TEXTS):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state["input_text"] = example
                st.rerun()
    
    # Text input area
    input_text = st.text_area(
        "Enter clinical/biomedical text:",
        value=st.session_state["input_text"],
        placeholder="Enter clinical notes, lab results, or biomedical text here...",
        height=150,
    )
    
    # Update session state when text changes
    st.session_state["input_text"] = input_text
    
    # Run inference button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        run_button = st.button("🔍 Extract Biomarkers", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state["input_text"] = ""
        st.rerun()
    
    # Run inference
    if run_button and input_text.strip():
        with st.spinner("Extracting biomarkers..."):
            results = run_inference(input_text, model, tokenizer)
        
        if results:
            st.divider()
            st.subheader("📋 Extraction Results")
            display_results(results)
            
            # Show raw JSON in expander
            with st.expander("📄 Raw JSON Output"):
                st.json(results)
        else:
            st.warning("Could not extract structured data. The model may have returned an unexpected format.")
    
    elif run_button and not input_text.strip():
        st.warning("Please enter some text to analyze.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.8em;">
        <p>Model: Qwen3.5-0.8B with LoRA adapter | Fine-tuned for biomarker extraction</p>
        <p>Powered by Unsloth | Deployed on Streamlit Cloud</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
