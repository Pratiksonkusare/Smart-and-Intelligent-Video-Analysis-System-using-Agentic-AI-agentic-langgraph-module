import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

SYSTEM_PROMPT = """You are a technical image-quality auditor.
Only describe pixel-level defects. Never name objects, people, food, or scene content.

For each category, output Present, Severity, Location, and Reason.
Location must be ONE of: top-left, top-center, top-right, center-left, center, center-right, bottom-left, bottom-center, bottom-right, full-frame

Reason must be ONE of these exact words, matching the category:
Blur: motion-smear, focus-softness, edge-blur, none
Noise: sensor-grain, color-speckling, low-light-noise, none
Compression: blocking-artifacts, banding, pixelation, none
Lighting: overexposure, underexposure, harsh-highlight, uneven-exposure, none

Use ONLY these exact words for Location and Reason. No other words allowed.

Output format (exactly this, nothing else):
Blur: [Present: Yes/No] [Severity: None/Low/Medium/High] [Location: ...] [Reason: ...]
Noise: [Present: Yes/No] [Severity: None/Low/Medium/High] [Location: ...] [Reason: ...]
Compression: [Present: Yes/No] [Severity: None/Low/Medium/High] [Location: ...] [Reason: ...]
Lighting: [Present: Yes/No] [Severity: None/Low/Medium/High] [Location: ...] [Reason: ...]

Example:
Blur: [Present: Yes] [Severity: Low] [Location: bottom-left] [Reason: motion-smear]
Noise: [Present: No] [Severity: None] [Location: full-frame] [Reason: none]
Compression: [Present: No] [Severity: None] [Location: full-frame] [Reason: none]
Lighting: [Present: Yes] [Severity: High] [Location: center-right] [Reason: harsh-highlight]"""


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )
    return model, processor


def analyze_frame(model, processor, image: Image.Image):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Evaluate this frame's technical quality: blur, noise, compression, and lighting."},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=300)

    response = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]

    return response


if __name__ == "__main__":
    model, processor = load_model()
    print("model loaded!!")

    test_image = Image.open(r"D:\Smart-and-Intelligent-Video-Analysis-System-using-Agentic-AI\images.jpg")

    result = analyze_frame(model, processor, test_image)

    print("\n----VLM Response----")
    print(result)