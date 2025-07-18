from flask import Flask, request, render_template, jsonify
import os

# Import your video analysis function
from utils import analyze_multiple_videos

# Ensure temp directory exists
os.makedirs('static/temp_videos', exist_ok=True)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        business = request.form['business']
        video_files = request.files.getlist('video')

        if not video_files:
            return jsonify({"error": "No video files uploaded."})

        filepaths = []
        for video in video_files[:5]:  # Limit to 5 videos
            save_path = os.path.join('static/temp_videos', video.filename)
            video.save(save_path)
            filepaths.append(save_path)
            print(f"[INFO] Saved {save_path}")

        print(f"[INFO] Analyzing {len(filepaths)} videos | Business: {business}")
        results = analyze_multiple_videos(filepaths, business)
        
        # No conversion needed since utils.py now returns a list directly
        print("[INFO] Analysis complete")
        return jsonify(results)
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)