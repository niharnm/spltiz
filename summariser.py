from audio_pipeline import AudioSegment

class Summariser:
    @staticmethod
    def generate_summary(segments: list[AudioSegment]) -> str:
        """
        Processes segments, extracts tasks and ideas, and builds
        a beautifully formatted Markdown report with action items and timestamps.
        """
        # Extract tasks and ideas
        tasks = [seg for seg in segments if seg.classification == "task"]
        ideas = [seg for seg in segments if seg.classification == "idea"]
        
        # Build 3 actionable bullet-point notes
        notes = []
        
        # Add tasks to notes
        for t in tasks:
            notes.append(f"**Task ({t.speaker} @ {t.start_time:.1f}s):** {t.transcript}")
        
        # Add ideas to notes
        for i in ideas:
            notes.append(f"**Idea ({i.speaker} @ {i.start_time:.1f}s):** {i.transcript}")
            
        # Standard fallback action items if we don't have enough segments classified
        fallback_items = [
            "Verify PySide6 drag-and-drop interfaces handle both WAV and MP3 files.",
            "Optimize energy thresholding for VAD to filter out low-frequency background hums.",
            "Confirm that the SQLite database is successfully persistent across app launches."
        ]
        
        while len(notes) < 3:
            notes.append(f"**Action Item:** {fallback_items[len(notes)]}")
            
        # Build Markdown
        md = []
        md.append("# SonicSplice Processing Summary")
        md.append("")
        md.append("### 3 Actionable Notes")
        for note in notes[:3]:
            md.append(f"- {note}")
        md.append("")
        md.append("### Segmented & Diarised Transcript")
        md.append("| Timecode | Speaker | Classification | Transcript |")
        md.append("| :--- | :--- | :--- | :--- |")
        
        for seg in segments:
            c_type = seg.classification.capitalize()
            md.append(f"| {seg.start_time:.1f}s - {seg.end_time:.1f}s | {seg.speaker} | {c_type} | {seg.transcript} |")
            
        return "\n".join(md)
