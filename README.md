
# Business Video Value Analyzer

## 📌 Description
This project automatically classifies uploaded business-related videos as **High Value** or **Low Value** using AI. The classification is based on:
- **Visual Clarity** (blurriness of frames)
- **Relevance** to the provided business description (via semantic similarity)

## 🛠 Features
- Frame extraction from videos
- Blurriness detection using Laplacian variance
- Video-level caption generation using **BLIP (BLIP-Large)**
- Semantic relevance scoring using **Sentence Transformers (all-mpnet-base-v2)**
- Verdict generation for business decision-making

## 📦 Tech Stack
- Python
- OpenCV
- HuggingFace Transformers
- Sentence Transformers
- PIL (for image processing)
- Flask (for web app, if used)

## 👩‍💻 How to Run
1. Place your video files in the desired folder.
2. Call `analyze_video(path, business_description)` from `utils.py`.
3. Output includes:
   - Captions
   - Relevance %
   - Blurriness info
   - Final Verdict