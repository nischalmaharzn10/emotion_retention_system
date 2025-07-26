from typing import Dict, List


def predict_churn(
    emotion_scores: Dict[str, float],
    history_emotions: List[Dict[str, float]]
) -> float:
    """
    Predict user churn risk based on current and historical emotion scores.

    :param emotion_scores: Latest emotion score dictionary.
    :param history_emotions: List of previous emotion score dictionaries.
    :return: Churn risk score between 0.0 and 1.0.
    """
    NEGATIVE = ['anger', 'sadness', 'fear', 'disgust']
    POSITIVE = ['joy', 'surprise', 'neutral']  # Treat 'neutral' as weakly positive

    # === Current base churn signal ===
    neg_score = sum(emotion_scores.get(e, 0.0) for e in NEGATIVE)
    pos_score = sum(emotion_scores.get(e, 0.0) for e in POSITIVE)
    base_risk = neg_score - 0.5 * pos_score  # Joy and neutral reduce churn risk

    # === History-based adjustment (last 6 entries, exponential decay) ===
    decay_weighted_neg = 0.0
    total_weight = 0.0

    for i, past_scores in enumerate(reversed(history_emotions[-6:])):
        weight = 0.9 ** i  # recent entries weigh more
        past_neg = sum(past_scores.get(e, 0.0) for e in NEGATIVE)
        decay_weighted_neg += weight * past_neg
        total_weight += weight

    history_penalty = decay_weighted_neg / total_weight if total_weight > 0 else 0.0

    # === Final churn score ===
    churn_risk = 0.5 * base_risk + 0.5 * history_penalty
    churn_risk = max(0.0, min(1.0, churn_risk))  # clamp between 0 and 1

    return round(churn_risk, 4)
