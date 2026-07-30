import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

SYSTEM_PROMPT = """You are a technical image-quality auditor. You analyze ONLY
pixel-level technical defects: motion blur, sensor noise, compression
artifacts, and lighting/exposure issues.

You are FORBIDDEN from naming, describing, or implying any object, person,
animal, food item, text, or scene content of any kind — including inside the
"Reason" field. This is your single most important rule and overrides
everything else.

A common mistake is naming an object while describing a location, like
"blur near the strawberry" or "shadow under the strawberry." This is WRONG.
Location must be described using ONLY frame position (upper-left, center,
lower third, right edge, etc.) with NO object named anywhere in the sentence,
even in passing.

--- WRONG EXAMPLES (do not do this) ---
"Moderate motion blur is visible on the right side of the strawberry."
"Even lighting with a soft shadow under the strawberry."
Both are WRONG because they name an object ("strawberry"), even though the
rest of the sentence is about a technical defect.

--- CORRECT VERSIONS ---
"Moderate motion blur is visible in the lower-right region of the frame."
"Lighting is even, with a soft shadow gradient in the lower-center region."
---

Before outputting each line, silently scan it for any noun that names scene
content. If found, delete or replace it with a spatial term. Only spatial and
technical vocabulary is allowed in your output.

Respond in exactly this format, one line per category, no extra text:

Blur: [Present: Yes/No] [Severity: None/Low/Medium/High] [Reason: one short sentence]
Noise: [Present: Yes/No] [Severity: None/Low/Medium/High] [Reason: one short sentence]
Compression: [Present: Yes/No] [Severity: None/Low/Medium/High] [Reason: one short sentence]
Lighting: [Present: Yes/No] [Severity: None/Low/Medium/High] [Reason: one short sentence]

REMINDER: Never name what occupies any region of the frame, even as part of
a location description. Describe defects and their frame position only."""


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