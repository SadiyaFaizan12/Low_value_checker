# Project Requirements Document

## 🎯 Objective
To help businesses automatically assess whether a video is valuable for their use, based on clarity and business relevance.

## 👨‍💻 Functionality
- **Frame Extraction:** Extract up to 3 diverse frames from each video.
- **Caption Generation:** Generate one representative caption using the middle frame.
- **Relevance Check:** Measure semantic similarity with the business description.
- **Blur Detection:** Determine how many frames are blurry.
- **Verdict Logic:** Combine clarity and relevance for final classification.

## 🧰 Dependencies
- `transformers==4.x`
- `sentence-transformers==2.x`
- `torch`
- `opencv-python`
- `Pillow`

## 📂 Files
- `utils.py`: All core functions
- `app.py`: (Optional Flask frontend)
- `requirements.txt`
