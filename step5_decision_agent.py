SEVERITY_RANK = {"None": 0, "Low": 1, "Medium": 2, "High": 3, "Unknown": 0}

ENHANCEABLE_CATEGORIES = ["Blur", "Noise", "Compression"]
SEVERITY_THRESHOLD  ="Medium"

def should_enhance(frame_result):
    """Looks at one frame's parsed VLM output and decides
    wheather it's bad enough to trigger autoamtic enhancement"""

    threshold_rank = SEVERITY_RANK[SEVERITY_THRESHOLD]
    triggered_by = []

    for category in ENHANCEABLE_CATEGORIES:
        info = frame_result[category]
        severity_rank = SEVERITY_RANK.get(info["severity"],0)

        if info["present"] and severity_rank>=threshold_rank:
            triggered_by.append(f"{category} ({info['severity']})")

    if triggered_by:
        reason = "Enhancement triggered by: " + ",".join(triggered_by)
        return True,reason

    return False, "No qualifying defects; enhancement skipped."    

if __name__ == "__main__":
    # quick standalone test using one of your real frame results
    sample_frame = {
        "Blur": {"present": True, "severity": "Medium", "reason": "..."},
        "Noise": {"present": False, "severity": "None", "reason": "..."},
        "Compression": {"present": True, "severity": "High", "reason": "..."},
        "Lighting": {"present": True, "severity": "High", "reason": "..."},
        "frame_number": 0,
    }

    decision, reason = should_enhance(sample_frame)
    print(f"Decision: {decision}")
    print(f"Reason: {reason}")