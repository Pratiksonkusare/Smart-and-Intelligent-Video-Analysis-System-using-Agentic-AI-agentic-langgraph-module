import cv2
import numpy as np

def compute_frame_metrics(pil_image):
    """
    Fast, classical CV metrics computed on the single frame
    No VLM cal -- this needs to run on all frames cheaply
    """

    rgb = np.array(pil_image)
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)

    # Blur : Laplacian variance, sharp edges create large second-derivative
    # spikes : a blurry image has smoothed-out edges, so this varience is low
    blur_varience = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness : simple mean pixel intensity
    brightness = gray.mean()
    

    noise_std = gray.std()

    contrast = gray.max().astype(float) - gray.min().astype(float)

    return {
        "blur_variance":float(blur_varience),
        "brightness":float(brightness),
        "noise_std":float(noise_std),
        "contrast":float(contrast)

    } 


def scan_all_frames(frames):
    """frames : the full list from extract_frames -- every frame, no interval
    skipping. Return each frame's metrics attached to its frame_number
    """

    scanned = []
    for frame in frames:
        metrics = compute_frame_metrics(frame["pil_image"])
        metrics["frame_number"] = frame["frame_number"]
        scanned.append(metrics)
    return scanned

# Detecting outlierssssss
def flag_suspicious_frames(scanned_metrics, blur_percentile=15, brightness_extreme=15,
                             absolute_blur_floor=80, absolute_dark_floor=40, absolute_bright_ceiling=235):
    blur_values = np.array([m["blur_variance"] for m in scanned_metrics])
    brightness_values = np.array([m["brightness"] for m in scanned_metrics])

    blur_cutoff = np.percentile(blur_values, blur_percentile)
    dark_cutoff = np.percentile(brightness_values, brightness_extreme)
    bright_cutoff = np.percentile(brightness_values, 100 - brightness_extreme)

    flagged = []
    for m in scanned_metrics:
        reasons = []

        # Relative outlier AND genuinely low in absolute terms -- not just
        # "worse than its neighbors" but actually objectively soft.
        if m["blur_variance"] <= blur_cutoff and m["blur_variance"] < absolute_blur_floor:
            reasons.append("low sharpness (relative + absolute)")

        if m["brightness"] <= dark_cutoff and m["brightness"] < absolute_dark_floor:
            reasons.append("unusually dark (relative + absolute)")

        if m["brightness"] >= bright_cutoff and m["brightness"] > absolute_bright_ceiling:
            reasons.append("unusually bright (relative + absolute)")

        if reasons:
            flagged.append({
                "frame_number": m["frame_number"],
                "reasons": reasons,
                "metrics": m,
            })

    return flagged

def collapse_to_representative_frames(flagged, min_gap=3):
    """
    Given flagged frames, groups consecutive/near-consecutive frame numbers
    into blocks (since they likely represent the same continuous defect
    event) and keeps only a few representatives per block instead of
    every frame -- cutting VLM calls without losing coverage of distinct
    defect events.
    """
    if not flagged:
        return []

    flagged_sorted = sorted(flagged, key=lambda f: f["frame_number"])
    blocks = []
    current_block = [flagged_sorted[0]]

    for f in flagged_sorted[1:]:
        if f["frame_number"] - current_block[-1]["frame_number"] <= min_gap:
            current_block.append(f)
        else:
            blocks.append(current_block)
            current_block = [f]
    blocks.append(current_block)

    representatives = []
    for block in blocks:
        # take first, middle, and last of each block -- enough to catch
        # a block that starts mild and gets worse, without sampling every frame
        if len(block) <= 3:
            representatives.extend(block)
        else:
            representatives.append(block[0])
            representatives.append(block[len(block) // 2])
            representatives.append(block[-1])

    return representatives

    




if __name__ == "__main__":
    import sys
    from Step1_video_basics import extract_frames

    video_path = sys.argv[1]

    frames = extract_frames(video_path,interval = 1)
    print(f"Total frames extracted : {len(frames)}")

    scanned = scan_all_frames(frames)
    flagged = flag_suspicious_frames(scanned)

    print(f"\nFlagged {len(flagged)} out of {len(frames)} frame as suspicious:")
    for f in flagged:
        print(f"Frame {f['frame_number']}: {f['reasons']}")
    



"""Laplacian varience -> the laplacian operator
highlights edges(rapid intensity changes). 
A sharp images has lot of string edges-> high varience. 
A blurry image has smoothed edges-> low varience.
"""

"""Brightness(mean) -> Averaging the pixel value, dark frames sit near 0
overexposed frames sit near 255, well exposed in between near (80-180)"""

'''Contrast (max−min) — a washed-out, low-contrast frame will have a 
small spread between its darkest and lightest pixel; 
a well-exposed frame usually spans a much wider range.'''
