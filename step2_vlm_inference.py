import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from step3_parser import parse_vlm_response
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
GENERIC_OBJECT_CATEGORIES = [
    # People
    "person", "man", "woman", "child", "kid", "baby", "people", "human",
    "face", "hand", "hair", "figure", "silhouette", "crowd",

    # Animals
    "animal", "dog", "cat", "bird", "fish", "horse", "cow", "pet",

    # Vehicles
    "car", "truck", "bike", "bicycle", "motorcycle", "bus", "vehicle",
    "train", "boat", "plane",

    # Food
    "food", "fruit", "vegetable", "apple", "banana", "orange", "strawberry",
    "plate", "bowl", "cup", "drink", "meal",

    # Furniture / indoor objects
    "table", "chair", "sofa", "couch", "bed", "desk", "shelf", "cabinet",
    "door", "window", "wall", "floor", "ceiling",

    # Electronics
    "phone", "laptop", "computer", "screen", "television", "tv", "camera",

    # Nature / outdoor
    "tree", "plant", "flower", "sky", "cloud", "grass", "mountain", "road",
    "building", "house",

    # Text / graphics
    "text", "logo", "sign", "label", "watermark", "number", "letter",

    # Clothing
    "shirt", "clothes", "hat", "shoe", "jacket",

    # Generic catch-alls
    "object", "item", "thing", "background", "foreground", "surface",
]

SYSTEM_PROMPT = """RULE: Never write the name of any object, person, animal, or food. This rule is more important than anything else in this prompt.

You are a technical image-quality auditor. Only describe blur, noise, compression, and lighting defects — nothing else.

Describe location using only: upper-left, upper-right, center, lower-left, lower-right, or full-frame. Never say what is located there.

Format (exactly, one line per category):
Blur: [Present: Yes/No] [Severity: None/Low/Medium/High] [Reason: ...]
Noise: [Present: Yes/No] [Severity: None/Low/Medium/High] [Reason: ...]
Compression: [Present: Yes/No] [Severity: None/Low/Medium/High] [Reason: ...]
Lighting: [Present: Yes/No] [Severity: None/Low/Medium/High] [Reason: ...]

Example:
Blur: [Present: Yes] [Severity: Low] [Reason: Slight smearing in the lower-left region.]
Noise: [Present: No] [Severity: None] [Reason: No visible grain.]
Compression: [Present: No] [Severity: None] [Reason: No artifacts observed.]
Lighting: [Present: Yes] [Severity: High] [Reason: The center-right region shows overexposure with harsh highlights and loss of detail.]

RULE (repeated): Never write the name of any object, person, animal, or food, anywhere in your response."""

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
    print("Model loaded successfully!!")
    
    test_image = Image.open("images.jpg")
    raw_result = analyze_frame(model,processor,test_image)

    print("\n----RAW VLM RESPONSE----")
    print(raw_result)
    print("\n---Response in Dictionary format---")
    parsed_result = parse_vlm_response(raw_result)
    import json
    print(json.dumps(parsed_result, indent = 2))