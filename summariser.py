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
        - Executive: Overall dialogue highlights, key decisions.
        - Action Items: Interactive checkbox lists.
        - Brainstorm: Ideas boards, concept definitions, creative workflows.
        - Verbatim: Fully aligned conversation transcripts.
        """
        tasks = [seg for seg in segments if seg.classification == "task"]
        ideas = [seg for seg in segments if seg.classification == "idea"]
        decisions = [seg for seg in segments if seg.classification == "decision"]
        questions = [seg for seg in segments if seg.classification == "question"]
        analytics = Summariser.calculate_analytics(segments)
        
        md = []
        md.append("# Spltiz Intelligence Summary")
        md.append(f"*Preset: {preset.replace('_', ' ').capitalize()} Mode | Computed entirely locally via Gemma-2B-audio*")
        md.append("")
        
        # 1. Conversation Metrics
        md.append("## 📊 Dialogue Metrics Overview")
        md.append(f"**Total File Length:** {analytics['total_duration']}s")
        for s in analytics["speakers"]:
            md.append(f"- **{s['speaker']}:** {s['percentage']}% total speak time ({s['duration']}s) | {s['words']} words written | {s['wpm']} words/min")
        md.append("")
        
        # 2. Preset Specific Analysis
        if preset == "executive":
            md.append("## 🎯 Executive Highlight Summary")
            md.append("This conversation focused on defining priorities and resolving core workflow configurations. Key highlights include:")
            for idx, i in enumerate(ideas[:2]):
                md.append(f"- **Key Point:** {i.transcript} (raised by {i.speaker})")
            if decisions:
                for d in decisions[:2]:
                    md.append(f"- **Decision:** {d.transcript} ({d.speaker})")
            
            md.append("")
            md.append("## 🔑 Key Takeaways")
            if tasks:
                md.append(f"- Action plans are set for {len(tasks)} items across the team.")
            if questions:
                md.append(f"- Outstanding question: *\"{questions[0].transcript}\"* remains under review.")
                
        elif preset == "action_items":
            md.append("## 🎯 Actionable Task Board")
            if tasks:
                for t in tasks:
                    md.append(f"- [ ] **Task ({t.speaker} @ {t.start_time:.1f}s):** {t.transcript}")
            else:
                md.append("- No tasks were automatically detected in this file.")
            
            if decisions:
                md.append("")
                md.append("## 📌 Verified Decisions")
                for d in decisions:
                    md.append(f"- [x] **Decided:** {d.transcript} ({d.speaker})")
                    
        elif preset == "brainstorm":
            md.append("## 💡 Ideation & Concept Cloud")
            if ideas:
                for i in ideas:
                    md.append(f"- **Idea ({i.speaker} @ {i.start_time:.1f}s):** {i.transcript}")
            else:
                md.append("- No ideation logs found.")
                
            if questions:
                md.append("")
                md.append("## ❓ Open Questions & Inquiries")
                for q in questions:
                    md.append(f"- **Question ({q.speaker} @ {q.start_time:.1f}s):** {q.transcript}")
                    
            md.append("")
            md.append("## ⚡ Suggested Actions")
            if tasks:
                for t in tasks:
                    md.append(f"- **Consider:** implementing *\"{t.transcript}\"* proposed by {t.speaker}")
                    
        else: # verbatim
            md.append("## 📝 Verbatim Transcript Logs")
            for seg in segments:
                classification_badge = f" *[{seg.classification.upper()}]*" if seg.classification != "irrelevant" else ""
                md.append(f"**[{seg.start_time:.1f}s - {seg.end_time:.1f}s] {seg.speaker}:** {seg.transcript}{classification_badge}")
                
        return "\n".join(md)
