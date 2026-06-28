from audio_pipeline import AudioSegment

class Summariser:
    @staticmethod
    def calculate_analytics(segments: list[AudioSegment]) -> dict:
        """
        Computes dialogue percentage distribution, word counts, and WPM rates
        to supply Otter-style conversational metrics.
        """
        total_duration = sum(seg.end_time - seg.start_time for seg in segments)
        if total_duration <= 0:
            total_duration = 1.0
            
        speaker_times = {}
        speaker_words = {}
        
        for seg in segments:
            duration = seg.end_time - seg.start_time
            speaker_times[seg.speaker] = speaker_times.get(seg.speaker, 0.0) + duration
            
            words = len(seg.transcript.split())
            speaker_words[seg.speaker] = speaker_words.get(seg.speaker, 0) + words
            
        analytics = []
        for speaker, time_spent in speaker_times.items():
            pct = (time_spent / total_duration) * 100
            words = speaker_words.get(speaker, 0)
            minutes = time_spent / 60.0
            wpm = words / minutes if minutes > 0 else 0.0
            
            analytics.append({
                "speaker": speaker,
                "percentage": round(pct, 1),
                "duration": round(time_spent, 1),
                "words": words,
                "wpm": round(wpm, 1)
            })
            
        return {
            "total_duration": round(total_duration, 1),
            "speakers": analytics
        }

    @staticmethod
    def generate_summary(segments: list[AudioSegment], preset: str = "executive") -> str:
        """
        Compiles structured insights based on preset rules:
        - Executive: Task tables, timelines, high-impact decisions.
        - Brainstorm: Ideas boards, concept definitions, creative workflows.
        - Verbatim: Fully aligned conversation transcripts.
        """
        tasks = [seg for seg in segments if seg.classification == "task"]
        ideas = [seg for seg in segments if seg.classification == "idea"]
        analytics = Summariser.calculate_analytics(segments)
        
        md = []
        md.append("# Spltiz Intelligence Summary")
        md.append(f"*Preset: {preset.capitalize()} Mode | Computed entirely locally via Gemma-2B-audio*")
        md.append("")
        
        # 1. Conversation Metrics
        md.append("## 📊 Dialogue Metrics Overview")
        md.append(f"**Total File Length:** {analytics['total_duration']}s")
        for s in analytics["speakers"]:
            md.append(f"- **{s['speaker']}:** {s['percentage']}% total speak time ({s['duration']}s) | {s['words']} words written | {s['wpm']} words/min")
        md.append("")
        
        # 2. Preset Specific Analysis
        if preset == "executive":
            md.append("## 🎯 Actionable Task Board")
            if tasks:
                for t in tasks:
                    md.append(f"- [ ] **Task ({t.speaker} @ {t.start_time:.1f}s):** {t.transcript}")
            else:
                md.append("- No tasks were automatically detected in this file.")
                
            md.append("")
            md.append("## 🔑 Key Decisions & Outlines")
            if ideas:
                for idx, i in enumerate(ideas[:3]):
                    md.append(f"{idx+1}. **Decision:** {i.transcript} ({i.speaker} at {i.start_time:.1f}s)")
            else:
                md.append("- No primary brainstorming decisions logged.")
                
        elif preset == "brainstorm":
            md.append("## 💡 Ideation & Concept Cloud")
            if ideas:
                for i in ideas:
                    md.append(f"- **Idea ({i.speaker} @ {i.start_time:.1f}s):** {i.transcript}")
            else:
                md.append("- No ideation logs found.")
            md.append("")
            md.append("## ⚡ Suggested Actions")
            if tasks:
                for t in tasks:
                    md.append(f"- **Consider:** implementing *\"{t.transcript}\"* proposed by {t.speaker}")
                    
        else: # verbatim
            md.append("## 📝 Verbatim Transcript Logs")
            for seg in segments:
                md.append(f"**[{seg.start_time:.1f}s - {seg.end_time:.1f}s] {seg.speaker}:** {seg.transcript}")
                
        return "\n".join(md)
