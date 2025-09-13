# src/export_to_onnx.py

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM

# ---------- SETTINGS ----------
MODEL_NAME = "distilgpt2"
ONNX_PATH = "models/distilgpt2.onnx"
# ------------------------------

def main():
    # Create the models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    print(f"Loading tokenizer and model from '{MODEL_NAME}'...")

    # Load the tokenizer and the PyTorch model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    print("Attempting to export the model to ONNX using optimum...")

    try:
        # Create an ORTModelForCausalLM from the pre-trained PyTorch model.
        # This will handle the ONNX export internally and more reliably.
        onnx_model = ORTModelForCausalLM.from_pretrained(
            model,
            export=True,
            from_transformers=True,
            feature="causal-lm",
            # Pass a dummy input to guide the export process
            # The optimum library handles the dynamic axes automatically.
            onnx_config=None,
        )

        # Save the exported model and tokenizer to a directory
        onnx_model.save_pretrained(ONNX_PATH)
        tokenizer.save_pretrained(ONNX_PATH)

        print(f"Export successful! The ONNX model and tokenizer are saved to '{ONNX_PATH}'.")

    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")

if __name__ == "__main__":
    main()
