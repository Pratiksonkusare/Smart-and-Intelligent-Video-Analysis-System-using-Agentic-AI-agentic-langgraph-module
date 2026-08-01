import cv2
import numpy as np
import torch
from torchvision.transforms.functional import rgb_to_grayscale
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from Step1_video_basics import extract_frames

WEIGHTS_PATH = "weights/RealESRGAN_x4plus.pth"


def load_enhancer():
    """
    Loads the real, pretrained Real-ESRGAN model.
    RRDBNet is the actual generator architecture (Residual-in-Residual
    Dense Block network) trained on millions of image pairs by the
    original researchers -- this is the real neural network.
    """
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23,
        num_grow_ch=32, scale=4,
    )

    enhancer = RealESRGANer(
        scale=4,
        model_path=WEIGHTS_PATH,
        model=model,
        tile=200,          # process in tiles to control VRAM usage
        tile_pad=10,
        pre_pad=0,
        half=True,          # fp16 -- same memory-saving idea as your VLM quantization
    )
    return enhancer


def enhance_frame(enhancer, pil_image):
    """
    Takes a PIL image, runs it through Real-ESRGAN, returns the enhanced
    result as a PIL image.
    """
    from PIL import Image

    rgb_array = np.array(pil_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)  # basicsr expects BGR

    output, _ = enhancer.enhance(bgr_array, outscale=4)

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output_rgb)


if __name__ == "__main__":
    from PIL import Image

    enhancer = load_enhancer()
    print("Real-ESRGAN loaded.")

    # test_image = Image.open("test_frame.jpg")
    # print(f"Original size: {test_image.size}")

    # enhanced = enhance_frame(enhancer, test_image)
    # print(f"Enhanced size: {enhanced.size}")

    # enhanced.save("test_frame_enhanced.jpg")
    # print("Saved enhanced frame to test_frame_enhanced.jpg")

    frames = extract_frames("test2.mp4",interval = 1)
    real_frame = frames[300]["pil_image"]
    real_frame.save("real_frame_original.jpg")

    enhanced = enhance_frame(enhancer, real_frame)
    enhanced.save("real_frame_enhanced.jpg")
    print("Saved real_frame_original.jpg and real_frame_enhanced.jpg")

    from step6_frame_scanner import compute_frame_metrics
    from PIL import Image

    original = Image.open("real_frame_original.jpg")
    enhanced = Image.open("real_frame_enhanced.jpg")

    enhanced_resized = enhanced.resize(original.size, Image.LANCZOS)

    print("Original metrics:", compute_frame_metrics(original))
    print("Enhanced metrics:", compute_frame_metrics(enhanced_resized))
