from typing import List, Dict

class ProctorAnalyzer:
    def __init__(self):
        # Base settings
        self.starting_score = 100
        self.review_threshold = 75
        self.auto_fail_threshold = 40
        
        # Penalties
        self.tab_switch_penalty = 25
        self.gaze_deviation_penalty = 5
        self.voice_overlap_penalty = 15
        
        # AI Cheating Delay (The "3-Second Rule")
        self.suspicious_delay_min = 3.5
        self.suspicious_delay_max = 6.0
        self.delay_penalty = 20

    def analyze_session(self, events: Dict) -> Dict:
        """
        Calculates the trust score based on session telemetry.
        Expected events payload:
        {
            "tab_switches": int,
            "gaze_deviations": int,
            "voice_overlap_events": int,
            "average_response_delay": float
        }
        """
        score = self.starting_score
        flags_triggered = []

        # 1. Tab Switching (High Severity)
        if events.get("tab_switches", 0) > 0:
            penalty = events["tab_switches"] * self.tab_switch_penalty
            score -= penalty
            flags_triggered.append(f"Tab switched {events['tab_switches']} times (-{penalty} pts)")

        # 2. Gaze Tracking (Medium Severity - allows for natural movement)
        # We allow 2 "free" looks away before penalizing
        gaze_count = events.get("gaze_deviations", 0)
        if gaze_count > 2:
            penalty = (gaze_count - 2) * self.gaze_deviation_penalty
            score -= penalty
            flags_triggered.append(f"Frequent off-screen gaze detected (-{penalty} pts)")

        # 3. Voice Overlap (Someone whispering an answer)
        if events.get("voice_overlap_events", 0) > 0:
            penalty = events["voice_overlap_events"] * self.voice_overlap_penalty
            score -= penalty
            flags_triggered.append(f"Multiple voices detected (-{penalty} pts)")

        # 4. AI Transcription Delay Pattern
        delay = events.get("average_response_delay", 0.0)
        if self.suspicious_delay_min <= delay <= self.suspicious_delay_max:
            score -= self.delay_penalty
            flags_triggered.append(f"Suspicious robotic response delay of {delay}s (-{self.delay_penalty} pts)")

        # Cap score at 0
        score = max(0, score)

        # Determine Verdict
        verdict = "PASS"
        if score <= self.auto_fail_threshold:
            verdict = "AUTO_FAIL"
        elif score <= self.review_threshold:
            verdict = "REQUIRE_HUMAN_REVIEW"

        return {
            "final_trust_score": score,
            "verdict": verdict,
            "flags": flags_triggered
        }

# Instantiate for use in main.py
proctor_service = ProctorAnalyzer()