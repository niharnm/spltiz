import os
import sqlite3
from flask import Flask, request, jsonify, render_template_string
from audio_pipeline import AudioPipeline
from summariser import Summariser

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the pipeline
pipeline = AudioPipeline()

# Premium HTML/JS Dashboard Template (Single Page Application)
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SonicSplice - Local-First Audio Intelligence</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet">
    <!-- MarkedJS for Markdown Rendering -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-color: #0b0b0e;
            --panel-bg: rgba(20, 20, 27, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-primary: #8a2be2;
            --accent-secondary: #00ffcc;
            --text-primary: #f3f3f6;
            --text-secondary: #a0a0ab;
            --glass-blur: blur(20px);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(138, 43, 226, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(0, 255, 204, 0.1) 0%, transparent 45%);
        }

        /* Glassmorphic Sidebar */
        aside {
            width: 320px;
            background: var(--panel-bg);
            backdrop-filter: var(--glass-blur);
            border-right: 1px solid var(--border-color);
            padding: 30px 20px;
            display: flex;
            flex-direction: column;
            gap: 25px;
            height: 100vh;
            position: sticky;
            top: 0;
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 20px;
            color: #000;
            box-shadow: 0 0 20px rgba(138, 43, 226, 0.4);
        }

        .logo-text {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #fff, #b5b5c9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .sidebar-title {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-secondary);
            margin-top: 15px;
        }

        .history-list {
            flex-grow: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 12px;
            padding-right: 5px;
        }

        /* Customize scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .history-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 15px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .history-item:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: var(--accent-primary);
            transform: translateY(-2px);
        }

        .history-item.active {
            background: linear-gradient(135deg, rgba(138, 43, 226, 0.15), rgba(0, 255, 204, 0.05));
            border-color: var(--accent-primary);
        }

        .history-filename {
            font-weight: 600;
            font-size: 14px;
            color: var(--text-primary);
            margin-bottom: 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .history-date {
            font-size: 11px;
            color: var(--text-secondary);
        }

        .clear-btn {
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 12px;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .clear-btn:hover {
            background: rgba(255, 0, 0, 0.1);
            color: #ff5e5e;
            border-color: rgba(255, 0, 0, 0.2);
        }

        /* Main Panel */
        main {
            flex-grow: 1;
            padding: 40px;
            display: flex;
            flex-direction: column;
            gap: 30px;
            max-width: 1200px;
            margin: 0 auto;
            width: calc(100% - 320px);
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .headline {
            font-size: 28px;
            font-weight: 800;
        }

        .badge {
            background: rgba(0, 255, 204, 0.1);
            color: var(--accent-secondary);
            border: 1px solid rgba(0, 255, 204, 0.2);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .badge::before {
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            background: var(--accent-secondary);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-secondary);
        }

        /* Ingestion Zone */
        .upload-card {
            background: var(--panel-bg);
            backdrop-filter: var(--glass-blur);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .upload-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            pointer-events: none;
            box-sizing: border-box;
            transition: all 0.4s ease;
        }

        .upload-card:hover {
            background: rgba(255, 255, 255, 0.02);
            transform: translateY(-2px);
        }

        .upload-card:hover::before {
            border-color: var(--accent-primary);
            box-shadow: inset 0 0 20px rgba(138, 43, 226, 0.15);
        }

        .upload-card.dragover::before {
            border-color: var(--accent-secondary);
            box-shadow: inset 0 0 20px rgba(0, 255, 204, 0.15);
        }

        .upload-icon {
            font-size: 48px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }

        .upload-title {
            font-size: 18px;
            font-weight: 600;
        }

        .upload-subtitle {
            font-size: 13px;
            color: var(--text-secondary);
        }

        #file-input {
            display: none;
        }

        /* Progress Display */
        .progress-container {
            display: none;
            width: 100%;
            flex-direction: column;
            gap: 10px;
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
        }

        .progress-status {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            font-weight: 600;
        }

        .progress-bar-bg {
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary));
            width: 0%;
            border-radius: 4px;
            transition: width 0.3s ease;
            position: relative;
        }

        /* Indeterminate state animation */
        .progress-bar-fill.indeterminate {
            width: 100%;
            animation: pulse-loading 1.5s infinite linear;
            background: linear-gradient(90deg, 
                var(--accent-primary) 0%, 
                var(--accent-secondary) 50%, 
                var(--accent-primary) 100%);
            background-size: 200% 100%;
        }

        @keyframes pulse-loading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        /* Timeline and Output Panel */
        .output-container {
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
        }

        .summary-card {
            background: var(--panel-bg);
            backdrop-filter: var(--glass-blur);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 30px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .card-header {
            font-size: 18px;
            font-weight: 700;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 15px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Markdown Styles for Summary */
        .markdown-body {
            font-size: 15px;
            line-height: 1.7;
            color: #d1d1db;
        }

        .markdown-body h1, .markdown-body h2, .markdown-body h3 {
            color: #fff;
            margin-top: 20px;
            margin-bottom: 10px;
            font-family: 'Space Grotesk', sans-serif;
        }

        .markdown-body ul {
            margin-left: 20px;
            margin-bottom: 15px;
        }

        .markdown-body li {
            margin-bottom: 8px;
        }

        .markdown-body table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }

        .markdown-body th {
            background: rgba(255, 255, 255, 0.05);
            text-align: left;
            padding: 10px 15px;
            font-weight: 600;
            border-bottom: 2px solid var(--border-color);
        }

        .markdown-body td {
            padding: 10px 15px;
            border-bottom: 1px solid var(--border-color);
        }

        .markdown-body tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }
    </style>
</head>
<body>

    <!-- Left Sidebar -->
    <aside>
        <div class="logo-container">
            <div class="logo-icon">S</div>
            <div class="logo-text">SonicSplice</div>
        </div>
        
        <div class="sidebar-title">Local Records</div>
        
        <div class="history-list" id="history-container">
            <!-- Loaded dynamically -->
        </div>

        <button class="clear-btn" onclick="clearHistory()">Clear History</button>
    </aside>

    <!-- Main Workspace -->
    <main>
        <header>
            <div class="headline">Audio Intelligence Dashboard</div>
            <div class="badge">Gemma-2B OpenVINO Accelerator</div>
        </header>

        <!-- Drag & Drop Ingestion -->
        <div class="upload-card" id="drop-zone" onclick="document.getElementById('file-input').click()">
            <div class="upload-icon">📥</div>
            <div class="upload-title">Drag & Drop WAV or MP3 audio file</div>
            <div class="upload-subtitle">Files are processed 100% locally and securely on your computer</div>
            <input type="file" id="file-input" accept=".mp3,.wav" onchange="handleFileSelect(event)">
        </div>

        <!-- Progress bar -->
        <div class="progress-container" id="progress-box">
            <div class="progress-status">
                <span id="progress-status-text">Uploading file...</span>
                <span>Active</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill indeterminate" id="progress-bar"></div>
            </div>
        </div>

        <!-- Document summary output -->
        <div class="output-container">
            <div class="summary-card">
                <div class="card-header">
                    <span>Generated Report & Transcript</span>
                </div>
                <div class="markdown-body" id="report-view">
                    <p style="color: var(--text-secondary); text-align: center; padding: 40px 0;">
                        Upload a voice memo or select a record from the history sidebar to begin analysis.
                    </p>
                </div>
            </div>
        </div>
    </main>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const progressBox = document.getElementById('progress-box');
        const progressStatusText = document.getElementById('progress-status-text');
        const reportView = document.getElementById('report-view');
        const historyContainer = document.getElementById('history-container');

        // Drag and Drop event handlers
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                uploadFile(files[0]);
            }
        });

        function handleFileSelect(event) {
            const files = event.target.files;
            if (files.length > 0) {
                uploadFile(files[0]);
            }
        }

        // Upload and process voice memo
        function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);

            dropZone.style.display = 'none';
            progressBox.style.display = 'flex';
            progressStatusText.innerText = "Segmenting audio and transcribing via Gemma-2B...";

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(res => {
                if (!res.ok) throw new Error("Processing failed");
                return res.json();
            })
            .then(data => {
                progressBox.style.display = 'none';
                dropZone.style.display = 'flex';
                
                // Render markdown summary output
                reportView.innerHTML = marked.parse(data.summary_markdown);
                loadHistory();
            })
            .catch(err => {
                progressBox.style.display = 'none';
                dropZone.style.display = 'flex';
                alert("Error processing audio: " + err.message);
            });
        }

        // History Management
        function loadHistory() {
            fetch('/history')
            .then(res => res.json())
            .then(data => {
                historyContainer.innerHTML = '';
                data.forEach((item, index) => {
                    const el = document.createElement('div');
                    el.className = 'history-item';
                    el.onclick = () => showHistoryItem(item.summary_markdown, el);
                    el.innerHTML = `
                        <div class="history-filename">${item.filename}</div>
                        <div class="history-date">${item.date_processed}</div>
                    `;
                    historyContainer.appendChild(el);
                });
            });
        }

        function showHistoryItem(markdown, el) {
            document.querySelectorAll('.history-item').forEach(item => item.classList.remove('active'));
            if (el) el.classList.add('active');
            reportView.innerHTML = marked.parse(markdown);
        }

        function clearHistory() {
            if (confirm("Are you sure you want to clear your local history?")) {
                fetch('/clear_history', { method: 'POST' })
                .then(() => {
                    reportView.innerHTML = `<p style="color: var(--text-secondary); text-align: center; padding: 40px 0;">Upload a voice memo to begin.</p>`;
                    loadHistory();
                });
            }
        }

        // Initialize history list on page load
        loadHistory();
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(INDEX_HTML)

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
        pipeline.save_to_history(file.filename, filepath, segments, summary_md)
        
        return jsonify({
            "filename": file.filename,
            "summary_markdown": summary_md
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
