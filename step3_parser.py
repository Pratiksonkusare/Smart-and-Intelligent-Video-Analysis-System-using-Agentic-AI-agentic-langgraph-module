import re

CATEGORIES = ["Blur", "Noise", "Compression", "Lighting"]


def parse_vlm_response(raw_text):
    """
    Converts the VLM's bracketed text output into a structured dict.
    Returns a dict shaped like:
        {"Blur": {"present": True, "severity": "Low", "reason": "..."}, ...}
    """
    result = {}

    for category in CATEGORIES:
        pattern = (
            rf"{category}:\s*\[Present:\s*(Yes|No)\]\s*"
            rf"\[Severity:\s*(\w+)\]\s*"
            rf"\[Reason:\s*(.*?)(?:\]|\n|$)"
    )

        match = re.search(pattern, raw_text)

        if match:
            present_str, severity_str, reason_str = match.groups()
            result[category] = {
                "present": present_str.strip().lower() == "yes",
                "severity": severity_str.strip(),
                "reason": reason_str.strip(),
            }
        else:
            # Defensive fallback: if the model didn't follow the format
            # for this category, don't crash -- just mark it unparseable.
            result[category] = {
                "present": False,
                "severity": "Unknown",
                "reason": "Could not parse model output for this category.",
            }

    return result


# if __name__ == "__main__":
#     sample_response = """Blur: [Present: Yes] [Severity: Low] [Reason: Slight motion blur visible in the lower-left region of the frame]
# Noise: [Present: No] [Severity: None] [Reason: No noticeable sensor noise in the image]
# Compression: [Present: No] [Severity: None] [Reason: No compression artifacts visible]
# Lighting: [Present: Yes] [Severity: Even] [Reason: Even lighting with a soft shadow in the lower-center region]"""

#     parsed = parse_vlm_response(sample_response)
#     import json
#     print(json.dumps(parsed, indent=2))