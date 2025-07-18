import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

# Load models once
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

MAX_VIDEO_UPLOAD = 5

def get_video_frames(video_path, max_frames=3):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total <= 0:
        return []
    
    step = max(1, total // max_frames)
    seen_hashes = set()

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            h = hash(frame.tobytes())
            if h not in seen_hashes:
                frames.append(frame)
                seen_hashes.add(h)
            if len(frames) >= max_frames:
                break
    cap.release()
    return frames

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < 100

def get_caption(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = caption_processor(image, return_tensors="pt")
    with torch.no_grad():
        out = caption_model.generate(**inputs, max_length=30, num_beams=5)
    return caption_processor.decode(out[0], skip_special_tokens=True)

def compute_relevance(captions, business_desc):
    business_emb = semantic_model.encode(business_desc, convert_to_tensor=True)
    scores = []

    for cap in captions:
        cap_emb = semantic_model.encode(cap, convert_to_tensor=True)
        score = util.pytorch_cos_sim(business_emb, cap_emb).item()
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return avg_score, scores

def analyze_video(path, business_desc):
    frames = get_video_frames(path)
    if not frames:
        return {
            "captions": [],
            "blurry": "No frames found",
            "relevance": "N/A",
            "verdict": "Error"
        }

    blur_count = sum([is_blurry(f) for f in frames])

    # ✅ Choose the middle frame for video-level caption
    middle_frame = frames[len(frames) // 2]
    video_caption = get_caption(middle_frame)

    # ✅ Treat single caption as a list to preserve output format
    captions = [video_caption]

    avg_relevance, all_scores = compute_relevance(captions, business_desc)

    low_quality = blur_count > len(frames) // 2
    low_relevance = avg_relevance < 0.5

    verdict = "Low Value" if low_quality or low_relevance else "High Value"

    return {
        "captions": captions,
        "relevance": f"{avg_relevance * 100:.1f}% relevant",
        "blurry": f"{blur_count}/{len(frames)} frames are blurry",
        "verdict": verdict
    }

def analyze_multiple_videos(video_paths, business_desc):
    results = []
    for path in video_paths[:MAX_VIDEO_UPLOAD]:
        filename = os.path.basename(path)
        try:
            result = analyze_video(path, business_desc)
            result["fileName"] = filename
            results.append(result)
        except Exception as e:
            results.append({
                "fileName": filename,
                "captions": [],
                "blurry": "-",
                "relevance": "-",
                "verdict": "Error",
                "error": str(e)
            })
    return results
