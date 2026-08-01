from Step1_video_basics import extract_frames
from step2_vlm_inference import load_model, analyze_frame
from step3_parser import parse_vlm_response
from step4_aggregator import aggregate_results
from step5_decision_agent import should_enhance
from step6_frame_scanner import scan_all_frames, flag_suspicious_frames,collapse_to_representative_frames


def run_full_pipeline(video_path):
    print("Stage 0: Extracting all frames...")
    all_frames = extract_frames(video_path, interval=1)
    print(f"Extracted {len(all_frames)} total frames.")

    print("\nStage 1: Cheap classical CV scan (all frames)...")
    scanned = scan_all_frames(all_frames)
    flagged = flag_suspicious_frames(scanned)
    flagged = collapse_to_representative_frames(flagged) 
    print(f"Flagged {len(flagged)} candidate frames out of {len(all_frames)} as Defective.")

    flagged_frame_numbers = {f["frame_number"] for f in flagged}
    candidate_frames = [f for f in all_frames if f["frame_number"] in flagged_frame_numbers]

    print("\nStage 2: VLM diagnosis (candidates only)...")
    model, processor = load_model()
    print("Model loaded successfully. Starting frame analysis...")

    vlm_results = []
    for idx, frame in enumerate(candidate_frames):
        print(f"Analyzing candidate {idx + 1}/{len(candidate_frames)} (frame {frame['frame_number']})...")
        raw = analyze_frame(model, processor, frame["pil_image"])
        parsed = parse_vlm_response(raw)
        parsed["frame_number"] = frame["frame_number"]
        vlm_results.append(parsed)

    print("\nStage 3: Decision + aggregation...")
    decisions = []
    for result in vlm_results:
        decision, reason = should_enhance(result)
        decisions.append({"frame_number": result["frame_number"], "decision": decision, "reason": reason})

    summary = aggregate_results(vlm_results, total_video_frames=len(all_frames))

    return {
        "total_frames": len(all_frames),
        "stage1_flagged_count": len(flagged),
        "vlm_results": vlm_results,
        "decisions": decisions,
        "video_summary": summary,
    }


if __name__ == "__main__":
    import sys
    import json

    video_path = sys.argv[1]
    result = run_full_pipeline(video_path)

    print(f"\n--- FINAL: {result['total_frames']} total frames, "
          f"{result['stage1_flagged_count']} sent to VLM ---")
    print(json.dumps(result["video_summary"], indent=2))