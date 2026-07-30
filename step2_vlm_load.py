import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

'''Here we have 2 billion parameters in this model,
and 16 or 32 bit (4-5gb) will take more space to get the parameter into the vram
so using quantization we are taking the same parameters/weights but
in 4 bits and they will be easy to load and will take less space'''

def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # shrinking parameters to 4 bits using nf4 method
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto"  # chooses GPU ram, and if it's full, uses CPU ram
    )
    return model, processor


if __name__ == "__main__":
    model, processor = load_model()
    print("Model loaded successfully")
    print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")