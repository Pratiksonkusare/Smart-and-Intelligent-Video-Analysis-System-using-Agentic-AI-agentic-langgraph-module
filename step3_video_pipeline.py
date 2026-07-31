from Step1_video_basics import extract_frames
from step2_vlm_inference import load_model,analyze_frame
from step3_parser import parse_vlm_response
from step4_aggregator import aggregate_results


def analyze_video(video_path,interval = 30):
    model,processor = load_model()
    print("Model Loaded!!")
    
    frames = extract_frames(video_path,interval)
    print(f"Extracted {len(frames)} frame to analyze.")

    all_result = []

    for frame in frames:
        frame_number = frame["frame_number"]
        pil_image = frame["pil_image"]

        print(f"Analyzing frame {frame_number}...")

        raw_response = analyze_frame(model,processor,pil_image)
        
        parsed = parse_vlm_response(raw_response)

        parsed["frame_number"] = frame_number
        all_result.append(parsed)

    return all_result

if __name__ == "__main__":
    import sys
    import json
    import torch
    print("CUDA available in this script:", torch.cuda.is_available())

    video_path = sys.argv[1]
    results = analyze_video(video_path, interval=30)

    print("\n---ALL FRAME RESULT---")
    print(json.dumps(results,indent=2)) 
    
    video_summary = aggregate_results(results)
    print(json.dumps(video_summary, indent=2))