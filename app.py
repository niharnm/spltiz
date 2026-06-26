import os
import sqlite3
from flask import Flask, request, jsonify, render_template
from audio_pipeline import AudioPipeline
from summariser import Summariser

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the pipeline
pipeline = AudioPipeline()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        # Run local VAD + diarization + inference
        segments = pipeline.segment_audio(filepath)
        for seg in segments:
            pipeline.run_inference(seg)
            
        summary_md = Summariser.generate_summary(segments)
        
        # Save run to database history
        history_id = pipeline.save_to_history(file.filename, filepath, segments, summary_md)
        
        return jsonify({
            "id": history_id,
            "filename": file.filename,
            "summary_markdown": summary_md
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/segments/<int:history_id>', methods=['GET'])
def get_segments(history_id):
    conn = sqlite3.connect(pipeline.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT start_time, end_time, speaker, transcript, classification
        FROM segments
        WHERE history_id = ?
        ORDER BY start_time ASC
    """, (history_id,))
    rows = cursor.fetchall()
    conn.close()
    
    segments = []
    for r in rows:
        segments.append({
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "speaker": r["speaker"],
            "transcript": r["transcript"],
            "classification": r["classification"]
        })
    return jsonify(segments)

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(pipeline.get_history())

@app.route('/clear_history', methods=['POST'])
def clear_history():
    conn = sqlite3.connect(pipeline.db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history")
    cursor.execute("DELETE FROM segments")
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
