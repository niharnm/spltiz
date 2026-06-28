import os
import sqlite3
import numpy as np
import pytest
from audio_pipeline import AudioPipeline, AudioSegment
from summariser import Summariser

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_history.db"
    return str(db_file)

def test_init_db(temp_db):
    pipeline = AudioPipeline(db_path=temp_db)
    assert os.path.exists(temp_db)
    
    # Check tables exist
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    assert "history" in tables
    assert "segments" in tables

def test_inference_and_segmentation_mock():
    # Setup test segments
    sr = 16000
    dummy_audio = np.zeros(sr * 2)  # 2 seconds of silence
    segment = AudioSegment(0.0, 2.0, "Speaker A", dummy_audio, sr)
    
    pipeline = AudioPipeline(db_path=":memory:")
    transcript, classification = pipeline.run_inference(segment)
    
    assert isinstance(transcript, str)
    assert len(transcript) > 0
    assert classification in ["idea", "task", "irrelevant"]

def test_summariser():
    # Setup standard list of segments
    sr = 16000
    dummy_audio = np.zeros(sr)
    
    seg1 = AudioSegment(0.0, 1.5, "Speaker A", dummy_audio, sr)
    seg1.transcript = "We need to fix the deployment scripts by tonight."
    seg1.classification = "task"
    
    seg2 = AudioSegment(1.5, 3.0, "Speaker B", dummy_audio, sr)
    seg2.transcript = "I think using Docker is the best choice here."
    seg2.classification = "idea"
    
    summary_md = Summariser.generate_summary([seg1, seg2])
    assert "# Spltiz Intelligence Summary" in summary_md
    assert "Dialogue Metrics Overview" in summary_md
    assert "Speaker A" in summary_md
    assert "Speaker B" in summary_md
    assert "Takeaways" in summary_md


def test_save_and_retrieve_history(temp_db):
    pipeline = AudioPipeline(db_path=temp_db)
    sr = 16000
    dummy_audio = np.zeros(sr)
    
    seg = AudioSegment(0.0, 2.0, "Speaker A", dummy_audio, sr)
    seg.transcript = "Please review the code."
    seg.classification = "task"
    
    summary_md = Summariser.generate_summary([seg])
    history_id = pipeline.save_to_history("test_audio.wav", "/path/to/test_audio.wav", [seg], summary_md)
    
    assert history_id > 0
    
    history = pipeline.get_history()
    assert len(history) == 1
    assert history[0]["filename"] == "test_audio.wav"
    assert history[0]["summary_markdown"] == summary_md

def test_rename_speaker(temp_db):
    pipeline = AudioPipeline(db_path=temp_db)
    sr = 16000
    dummy_audio = np.zeros(sr)
    
    seg = AudioSegment(0.0, 2.0, "Speaker A", dummy_audio, sr)
    seg.transcript = "Let's review this segment."
    seg.classification = "idea"
    
    summary_md = Summariser.generate_summary([seg])
    history_id = pipeline.save_to_history("test.wav", "/path/to/test.wav", [seg], summary_md)
    
    # Rename "Speaker A" to "Nihar"
    pipeline.rename_speaker(history_id, "Speaker A", "Nihar")
    
    # Check if updated in database
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT speaker FROM segments WHERE history_id = ?", (history_id,))
    speaker_val = cursor.fetchone()[0]
    
    cursor.execute("SELECT summary_markdown FROM history WHERE id = ?", (history_id,))
    updated_md = cursor.fetchone()[0]
    conn.close()
    
    assert speaker_val == "Nihar"
    assert "Nihar" in updated_md
    assert "Speaker A" not in updated_md

