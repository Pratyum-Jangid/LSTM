import math


class ThreatAssessment:

    def assess(
        self,
        avg_speed,
        max_speed,
        avg_acceleration,
        max_acceleration,
        track_duration,
    ):
        """
        Calculate threat score for ONE tracked drone.

        This is currently a heuristic prototype.
        Speed and acceleration are image-space quantities
        (pixels/frame), not physical m/s.
        """

        score = 0
        reasons = []

        # ---------------------------------------------------------
        # 1. Average speed
        # ---------------------------------------------------------

        if avg_speed >= 6:
            score += 20
            reasons.append("High average motion.")

        elif avg_speed >= 3:
            score += 10
            reasons.append("Moderate average motion.")

        # ---------------------------------------------------------
        # 2. Peak speed
        # ---------------------------------------------------------

        if max_speed >= 80:
            score += 25
            reasons.append("Very high peak motion.")

        elif max_speed >= 30:
            score += 15
            reasons.append("High peak motion.")

        elif max_speed >= 10:
            score += 5
            reasons.append("Moderate peak motion.")

        # ---------------------------------------------------------
        # 3. Average acceleration
        # ---------------------------------------------------------

        if abs(avg_acceleration) >= 2:
            score += 10
            reasons.append("Sustained acceleration.")

        # ---------------------------------------------------------
        # 4. Peak acceleration
        # ---------------------------------------------------------

        if abs(max_acceleration) >= 80:
            score += 20
            reasons.append("Aggressive maneuver detected.")

        elif abs(max_acceleration) >= 20:
            score += 10
            reasons.append("Rapid maneuver detected.")

        # ---------------------------------------------------------
        # 5. Track persistence
        # ---------------------------------------------------------

        if track_duration >= 300:
            score += 15
            reasons.append("Persistent tracked object.")

        elif track_duration >= 150:
            score += 8
            reasons.append("Object tracked for significant duration.")

        # ---------------------------------------------------------
        # Clamp score
        # ---------------------------------------------------------

        score = min(score, 100)

        # ---------------------------------------------------------
        # Threat level
        # ---------------------------------------------------------

        if score >= 70:
            level = "HIGH"

        elif score >= 45:
            level = "MEDIUM"

        elif score >= 20:
            level = "LOW"

        else:
            level = "NO THREAT"

        return {
            "score": score,
            "level": level,
            "reasons": reasons,
        }