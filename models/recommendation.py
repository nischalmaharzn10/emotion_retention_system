from typing import TypedDict, Literal, Dict


class ActionRecommendation(TypedDict):
    message: str
    code: Literal['ESCALATE', 'DEFUSE', 'SUPPORT', 'CHECKIN', 'ENGAGE', 'OBSERVE']


class RecommendationConfig(TypedDict):
    ESCALATE_CUTOFF: float
    DEFUSE_CUTOFF: float
    CHECKIN_CUTOFF: float


# === Default Threshold Configuration ===
default_config: RecommendationConfig = {
    "ESCALATE_CUTOFF": 0.75,
    "DEFUSE_CUTOFF": 0.60,
    "CHECKIN_CUTOFF": 0.40,
}


def recommend_action(churn_risk: float, emotion_scores: Dict[str, float]) -> ActionRecommendation:
    """
    Determines the appropriate action recommendation based on churn risk and dominant emotion.
    
    :param churn_risk: A float between 0.0 and 1.0 indicating likelihood of user churn.
    :param emotion_scores: A dictionary of emotion names mapped to their confidence scores.
    :return: A recommendation with an actionable message and a code.
    """
    if not emotion_scores:
        return {"message": "Insufficient data", "code": "OBSERVE"}

    dominant_emotion = max(emotion_scores, key=lambda k: emotion_scores.get(k, 0.0))

    # High churn risk — immediate escalation
    if churn_risk >= default_config["ESCALATE_CUTOFF"]:
        return {"message": "Escalate to human support", "code": "ESCALATE"}

    # Medium-high churn risk — emotional mitigation
    if churn_risk >= default_config["DEFUSE_CUTOFF"]:
        if dominant_emotion in {"anger", "disgust"}:
            return {"message": "De-escalate with empathy", "code": "DEFUSE"}
        elif dominant_emotion in {"fear", "sadness"}:
            return {"message": "Offer emotional reassurance", "code": "SUPPORT"}

    # Moderate churn risk — proactive support
    if churn_risk >= default_config["CHECKIN_CUTOFF"]:
        return {"message": "Proactively check in and offer help", "code": "CHECKIN"}

    # Low churn — reinforce good vibes
    if dominant_emotion == "joy":
        return {"message": "Reinforce positive interaction", "code": "ENGAGE"}

    # All else — passive monitoring
    return {"message": "Monitor sentiment passively", "code": "OBSERVE"}
