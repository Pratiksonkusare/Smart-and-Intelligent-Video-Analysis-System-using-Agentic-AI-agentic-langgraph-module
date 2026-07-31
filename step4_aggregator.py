SEVERITY_RANK = {"None":0,"Low":1,"Medium":2,"High":3,"Unknown":0}
CATEGORIES = ["Blur","Noise","Compression","Lighting"]

def aggregate_results(frame_results, total_video_frames=None):
    """
    total_video_frames: the TRUE total frame count of the video (e.g. 1116),
    not just len(frame_results) -- since frame_results may only be the
    VLM-checked candidates after Stage 1 filtering.
    """
    summary = {}
    denominator = total_video_frames if total_video_frames else len(frame_results)

    for category in CATEGORIES:
        frames_present = []
        worst_severity = "None"
        worst_rank = 0

        for frame in frame_results:
            info = frame[category]
            if info["present"]:
                frames_present.append(frame["frame_number"])
            rank = SEVERITY_RANK.get(info["severity"], 0)
            if rank > worst_rank:
                worst_rank = rank
                worst_severity = info["severity"]

        summary[category] = {
            "worst_severity": worst_severity,
            "frames_affected": len(frames_present),
            "total_frames_in_video": denominator,
            "frames_checked_by_vlm": len(frame_results),
            "percent_of_video_affected": round(100 * len(frames_present) / denominator, 1),
            "affected_frame_numbers": frames_present,
        }

    return summary

if __name__ == "__main__":
        fake_results = [
        {"Blur": {"present": True, "severity": "Medium", "reason": "..."},
         "Noise": {"present": False, "severity": "None", "reason": "..."},
         "Compression": {"present": False, "severity": "None", "reason": "..."},
         "Lighting": {"present": True, "severity": "High", "reason": "..."},
         "frame_number": 0},
        {"Blur": {"present": True, "severity": "Low", "reason": "..."},
         "Noise": {"present": True, "severity": "Low", "reason": "..."},
         "Compression": {"present": False, "severity": "None", "reason": "..."},
         "Lighting": {"present": True, "severity": "High", "reason": "..."},
         "frame_number": 30},
        ]
        import json
        print(json.dumps(aggregate_results(fake_results),indent=2))
            

