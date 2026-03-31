"""
Biomarker Extraction Streamlit App
Uses Qwen3.5-0.8B merged model via HuggingFace Inference API

This app makes API calls to HuggingFace's inference endpoint.
No local model loading - everything runs remotely.
"""

import streamlit as st
import json
import re
from huggingface_hub import InferenceClient

# Constants
MODEL_NAME = "biolytai123/biomarker-qwen3.5-0.8b-merged-v2"

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


def extract_json_from_response(response: str) -> dict:
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


def run_inference(text: str, hf_token: str) -> dict:
    """Run inference via HuggingFace Inference API."""
    client = InferenceClient(model=MODEL_NAME, token=hf_token)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract biomarkers from this clinical text: {text}"}
    ]
    
    try:
        response = client.chat_completion(
            messages,
            max_tokens=512,
            temperature=0.1,
        )
        
        # Extract the response text
        if hasattr(response, 'choices') and response.choices:
            response_text = response.choices[0].message.content
        elif isinstance(response, dict) and 'choices' in response:
            response_text = response['choices'][0]['message']['content']
        else:
            response_text = str(response)
        
        # Parse JSON from response
        result = extract_json_from_response(response_text)
        return result if result else {"raw": response_text}
        
    except Exception as e:
        return {"error": str(e)}


def display_results(results: dict):
    """Display the extracted biomarkers in a nice format."""
    if not results:
        st.warning("No biomarkers could be extracted from the text.")
        return
    
    if "error" in results:
        st.error(f"Error: {results['error']}")
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
    
    **Note:** This app uses HuggingFace Inference API — no local model needed!
    """)
    
    # Input for HF token
    st.sidebar.header("⚙️ Configuration")
    hf_token = st.sidebar.text_input(
        "HuggingFace Token",
        type="password",
        help="Your HF token for inference. Get it from https://huggingface.co/settings/tokens"
    )
    
    if not hf_token:
        st.info("👈 Please enter your HuggingFace token in the sidebar to start!")
        st.stop()
    
    # Initialize session state
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    
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
        with st.spinner("Extracting biomarkers via HuggingFace API..."):
            results = run_inference(input_text, hf_token)
        
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
        <p>Model: Qwen3.5-0.8B merged with LoRA adapter | Hosted on HuggingFace Inference API</p>
        <p>Deployed on Streamlit Cloud</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
