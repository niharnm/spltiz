import os
import sys
from PySide6.QtCore import Qt, QMimeData, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QTextBrowser, QProgressBar,
    QFileDialog, QSplitter, QMessageBox, QFrame
)
from audio_pipeline import AudioPipeline
from summariser import Summariser

class AudioWorker(QThread):
    finished = Signal(list, str)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, pipeline: AudioPipeline, filepath: str):
        super().__init__()
        self.pipeline = pipeline
        self.filepath = filepath

    def run(self):
        try:
            filename = os.path.basename(self.filepath)
            self.progress.emit("Loading and segmenting audio (VAD)...")
            segments = self.pipeline.segment_audio(self.filepath)
            
            if not segments:
                self.error.emit("No audio segments detected.")
                return

            total = len(segments)
            for idx, seg in enumerate(segments):
                self.progress.emit(f"Transcribing segment {idx+1}/{total} (Gemma-2B)...")
                self.pipeline.run_inference(seg)

            self.progress.emit("Compiling summary and actionable tasks...")
            summary_md = Summariser.generate_summary(segments)
            
            # Save run to SQLite database
            self.pipeline.save_to_history(filename, self.filepath, segments, summary_md)
            
            self.finished.emit(segments, summary_md)
        except Exception as e:
            self.error.emit(str(e))


class DropArea(QFrame):
    fileDropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.setLineWidth(2)
        self.setObjectName("DropArea")
        
        layout = QVBoxLayout(self)
        self.label = QLabel("Drag & Drop .mp3 or .wav file here\n(or click to browse)")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 16px; font-weight: bold; color: #a5a5b1;")
        layout.addWidget(self.label)
        
        self.setStyleSheet("""
            QFrame#DropArea {
                border: 2px dashed #5d5d6a;
                border-radius: 12px;
                background-color: #1a1a20;
            }
            QFrame#DropArea:hover {
                border-color: #7c4dff;
                background-color: #211f2d;
            }
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and (urls[0].toLocalFile().lower().endswith(('.mp3', '.wav'))):
                event.acceptProposedAction()
                self.setStyleSheet("""
                    QFrame#DropArea {
                        border: 2px dashed #7c4dff;
                        background-color: #27213d;
                    }
                """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame#DropArea {
                border: 2px dashed #5d5d6a;
                background-color: #1a1a20;
            }
        """)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            filepath = event.mimeData().urls()[0].toLocalFile()
            self.setStyleSheet("""
                QFrame#DropArea {
                    border: 2px dashed #5d5d6a;
                    background-color: #1a1a20;
                }
            """)
            self.fileDropped.emit(filepath)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Select Audio File", "", "Audio Files (*.mp3 *.wav)"
            )
            if filepath:
                self.fileDropped.emit(filepath)


class SpltizApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pipeline = AudioPipeline()
        self.init_ui()
        self.load_history()

    def init_ui(self):
        self.setWindowTitle("Spltiz - Local-First Intelligent Audio Splicer")
        self.resize(1100, 700)
        self.setMinimumSize(850, 600)
        
        # Modern Premium Dark Palette Styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f13;
            }
            QWidget {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                color: #e1e1e6;
            }
            QListWidget {
                background-color: #15151b;
                border: 1px solid #2a2a35;
                border-radius: 8px;
                padding: 5px;
            }
            QListWidget::item {
                background-color: #1f1f2a;
                border-radius: 6px;
                padding: 10px;
                margin-bottom: 5px;
                color: #e1e1e6;
            }
            QListWidget::item:hover {
                background-color: #272737;
            }
            QListWidget::item:selected {
                background-color: #7c4dff;
                color: #ffffff;
            }
            QTextBrowser {
                background-color: #15151b;
                border: 1px solid #2a2a35;
                border-radius: 8px;
                padding: 15px;
                font-size: 14px;
                line-height: 1.6;
            }
            QPushButton {
                background-color: #7c4dff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #651fff;
            }
            QProgressBar {
                border: 1px solid #2a2a35;
                border-radius: 6px;
                background-color: #15151b;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #7c4dff;
                border-radius: 5px;
            }
            QLabel#Title {
                font-size: 20px;
                font-weight: bold;
                color: #ffffff;
            }
            QLabel#SidebarTitle {
                font-size: 14px;
                font-weight: bold;
                color: #a5a5b1;
            }
        """)

        # Main Layout using Splitter
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # 1. Left Sidebar: History Panel
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        
        sb_title = QLabel("Transcription History")
        sb_title.setObjectName("SidebarTitle")
        sidebar_layout.addWidget(sb_title)
        
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_history_item_clicked)
        sidebar_layout.addWidget(self.history_list)
        
        clear_btn = QPushButton("Clear History")
        clear_btn.setStyleSheet("background-color: #3f3f4a; color: #e1e1e6;")
        clear_btn.clicked.connect(self.clear_history)
        sidebar_layout.addWidget(clear_btn)
        
        # 2. Right Panel: Workspace Panel
        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(5, 5, 5, 5)
        
        # Header / Title block
        header = QHBoxLayout()
        title_lbl = QLabel("Spltiz")
        title_lbl.setObjectName("Title")
        header.addWidget(title_lbl)
        
        # Badge for Gemma model acceleration
        model_badge = QLabel("Gemma-2B OpenVINO Accelerator Active")
        model_badge.setStyleSheet("color: #00e676; background-color: #1b3a24; border-radius: 4px; padding: 4px 8px; font-size: 11px;")
        header.addStretch()
        header.addWidget(model_badge)
        workspace_layout.addLayout(header)

        # Drag and Drop Area
        self.drop_area = DropArea()
        self.drop_area.fileDropped.connect(self.start_processing)
        workspace_layout.addWidget(self.drop_area)
        
        # Processing Progress Tracker
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        workspace_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #a5a5b1; font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        workspace_layout.addWidget(self.status_label)
        
        # Output Markdown Display Area
        self.output_browser = QTextBrowser()
        self.output_browser.setPlaceholderText("Upload an audio file to see the transcript, speaker diarisation, and notes.")
        workspace_layout.addWidget(self.output_browser)
        
        # Set proportions: Sidebar 25%, Workspace 75%
        splitter.addWidget(sidebar)
        splitter.addWidget(workspace)
        splitter.setSizes([250, 800])
        
        # Load pipeline model (non-blocking notification)
        self.pipeline.load_model()

    def load_history(self):
        self.history_list.clear()
        history_items = self.pipeline.get_history()
        for item in history_items:
            # We list filename & date in list widget items
            display_text = f"{item['filename']}\n({item['date_processed']})"
            self.history_list.addItem(display_text)

    def on_history_item_clicked(self, item):
        # Retrieve original history items
        history_items = self.pipeline.get_history()
        idx = self.history_list.row(item)
        if idx >= 0 and idx < len(history_items):
            markdown_content = history_items[idx]["summary_markdown"]
            self.output_browser.setMarkdown(markdown_content)

    def clear_history(self):
        import sqlite3
        conn = sqlite3.connect(self.pipeline.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history")
        cursor.execute("DELETE FROM segments")
        conn.commit()
        conn.close()
        self.load_history()
        self.output_browser.clear()

    def start_processing(self, filepath: str):
        # Disable drop area and show progress bar
        self.drop_area.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Pulse indicator
        
        self.worker = AudioWorker(self.pipeline, filepath)
        self.worker.progress.connect(self.update_progress_status)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def update_progress_status(self, text: str):
        self.status_label.setText(text)

    def on_processing_finished(self, segments, summary_md):
        self.drop_area.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Processing completed successfully!")
        
        self.output_browser.setMarkdown(summary_md)
        self.load_history()

    def on_processing_error(self, err_msg: str):
        self.drop_area.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error processing file.")
        QMessageBox.critical(self, "Error", f"An error occurred during audio processing:\n{err_msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpltizApp()
    window.show()
    sys.exit(app.exec())
