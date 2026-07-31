SEVERITY_RANK = {"None":0,"Low":1,"Medium":2,"High":3,"Unknown":0}
CATEGORIES = ["Blur","Noise","Compression","Lighting"]

def aggregate_results(frame_results):
    """
    Take the list of per-frame parsed dicts and collapes them into
    one video-level summary per defect category"""

    summary={}

    for category in CATEGORIES:
        frame_present=[]
        worst_severity = "None"
        worst_rank = 0

        for frame in frame_results:
            info = frame[category]
            if info["present"]:
                frame_present.append(frame["frame_number"])

            rank = SEVERITY_RANK.get(info["severity"],0)
            if rank>worst_rank:
                worst_rank = rank
                worst_severity = info["severity"]
        total_frames = len(frame_results)
        summary[category] = {
            "worst_severity" : worst_severity,
            "frame_affected":len(frame_present),
            "total_frames":total_frames,
            "percent_affected":round(100*len(frame_present)/total_frames,1),
            "affected_frame_numbers" : frame_present
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
            

