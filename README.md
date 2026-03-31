# Biomarker Extraction Streamlit App

A production-ready Streamlit application that extracts biomarkers, medical entities, and clinical measurements from biomedical text using the Qwen3.5-0.8B model fine-tuned with LoRA for biomarker extraction.

## Model

- **Base Model**: Qwen/Qwen3.5-0.8B
- **LoRA Adapter**: Shubh-0789/biomarker-qwen3.5-0.8b-lora-v2
- **Loading**: bf16 (NOT 4-bit quantization)

## Features

- Clean, intuitive Streamlit UI
- Extracts biomarkers, values, medical entities, and conditions
- Example clinical texts for testing
- JSON output for programmatic use
- Error handling and loading states

## Deployment

This app is designed for deployment on Streamlit Cloud.

### Deploy to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New app" → "Deploy an existing app"
4. Select your forked repository
5. Set the main file path to `app.py`
6. Click "Deploy!"

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/biomarker-streamlit.git
cd biomarker-streamlit

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## API Reference

### Input
Enter any clinical or biomedical text containing:
- Lab results
- Blood test values
- Tumor markers
- Vital signs
- Medical conditions

### Output
The model returns structured JSON with:
- `biomarkers`: List of biomarker names
- `values`: Measured values with units
- `entities`: Other medical/clinical entities
- `conditions`: Diseases or conditions mentioned

## Example Usage

```
Patient shows elevated CRP levels at 12.5 mg/L indicating possible inflammation. 
Blood glucose measured at 126 mg/dL suggesting pre-diabetic condition. 
LDL cholesterol 145 mg/dL.
```

Expected output:
```json
{
  "biomarkers": ["CRP", "Blood glucose", "LDL cholesterol"],
  "values": ["12.5 mg/L", "126 mg/dL", "145 mg/dL"],
  "entities": ["inflammation", "pre-diabetic condition"],
  "conditions": []
}
```

## Requirements

- streamlit >= 1.28.0
- torch >= 2.0.0
- transformers >= 4.40.0
- unsloth >= 2024.4.0
- peft >= 0.10.0
- accelerate >= 0.27.0
- bitsandbytes >= 0.41.0

## License

MIT License

## Author

Fine-tuned model by Shubh-0789
