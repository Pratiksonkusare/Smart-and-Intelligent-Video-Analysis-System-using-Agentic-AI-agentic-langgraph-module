import cv2

def get_video_metadata(video_path):

    #cv2.VideoCapture opens a video file

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")


    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps>0 else 0

    cap.release()

    return {
    "fps":fps,
    "frame_count":frame_count,
    "width":width,
    "height":height,
    "Duration":duration
    }
    

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    meta = get_video_metadata(path)
    print(meta)