import logging
from typing import TypedDict, List, Dict, Union, Optional, Any, cast
from datetime import datetime

from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage

from models.emotion_model import EmotionDetector
from models.churn_predictor import predict_churn
from models.recommendation import recommend_action, ActionRecommendation
from memory.memory_manager import MemoryManager


# === Configure Logger ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# === TypedDicts ===
class Message(TypedDict):
    role: str
    content: str


class WorkflowState(TypedDict, total=False):
    input: str
    emotion_scores: Dict[str, float]
    history: List[Message]
    churn_risk: float
    recommendation: Union[str, ActionRecommendation]


# === Global Instances ===
emotion_model = EmotionDetector()
memory = MemoryManager()


# === Helper ===
def make_message(role: str, content: str) -> Message:
    return {"role": role, "content": content}


# === Node Functions ===
def node_emotion(state: WorkflowState) -> WorkflowState:
    input_text: Optional[str] = state.get("input")

    if not input_text or not isinstance(input_text, str):
        logger.error("Invalid input at node_emotion")
        raise ValueError("Missing or invalid 'input'")

    emotion = emotion_model.detect(input_text) or {"neutral": 1.0}
    logger.debug(f"[node_emotion] {emotion}")

    new_state = state.copy()
    new_state["emotion_scores"] = emotion
    return new_state


def node_memory(state: WorkflowState) -> WorkflowState:
    try:
        input_text = state.get("input", "").strip()
        rec = state.get("recommendation")

        logger.debug(f"[node_memory] Recommendation: {rec}")

        if not input_text:
            logger.warning("No input text to save")
            return state

        if not rec:
            logger.warning("No recommendation in state, skipping memory save")
            return state

        # Determine AI response
        ai_response = rec.get("message", "") if isinstance(rec, dict) else str(rec or "")
        if not ai_response.strip():
            logger.warning("Empty AI response; skipping memory save")
            return state

        # Save entry to memory
        memory.add(input_text, ai_response, state.get("emotion_scores"))

        # Load history
        raw_history = memory.get_context()
        typed_history = [
            make_message(item["role"], item["content"])
            for item in raw_history if item["content"].strip()
        ][:20]

        new_state = state.copy()
        new_state["history"] = typed_history
        return new_state

    except Exception:
        logger.exception("Error in node_memory")
        new_state = state.copy()
        new_state["history"] = []
        return new_state


def node_churn(state: WorkflowState) -> WorkflowState:
    emotion_scores = state.get("emotion_scores")
    if not isinstance(emotion_scores, dict):
        logger.error("Missing or invalid emotion_scores at node_churn")
        raise ValueError("Invalid emotion_scores")

    churn_risk = predict_churn(emotion_scores, [])
    if not isinstance(churn_risk, float):
        logger.error("predict_churn returned invalid type")
        raise ValueError("predict_churn must return float")

    new_state = state.copy()
    new_state["churn_risk"] = churn_risk
    return new_state


def node_recommend(state: WorkflowState) -> WorkflowState:
    churn = state.get("churn_risk")
    emotions = state.get("emotion_scores")

    if churn is None or not isinstance(churn, float) or not isinstance(emotions, dict):
        logger.error("Invalid inputs at node_recommend")
        raise ValueError("Invalid state for recommendation")

    rec = recommend_action(churn_risk=churn, emotion_scores=emotions)

    new_state = state.copy()
    new_state["recommendation"] = rec
    logger.debug(f"[node_recommend] {rec}")
    return new_state


def node_response(state: WorkflowState) -> AIMessage:
    emotion_scores: Dict[str, float] = state.get("emotion_scores", {})
    churn_risk: float = state.get("churn_risk", 0.0)
    recommendation_raw: Union[Dict[str, Any], str, ActionRecommendation] = state.get("recommendation", {})

    top_emotion = max(emotion_scores, key=lambda k: emotion_scores[k]) if emotion_scores else "neutral"

    message = (
        recommendation_raw.get("message", "No recommendation available.")
        if isinstance(recommendation_raw, dict)
        else str(recommendation_raw)
    )

    response_text = (
        f"Based on your emotional state (mostly {top_emotion}) "
        f"and a churn risk of {churn_risk:.2f}, we recommend: {message}"
    )

    return AIMessage(content=response_text)


# === Workflow Builder ===
def build_workflow() -> Any:
    g = StateGraph(WorkflowState)

    g.add_node("emotion", node_emotion)
    g.add_node("churn", node_churn)
    g.add_node("recommend", node_recommend)
    g.add_node("memory", node_memory)
    # g.add_node("response", node_response) 

    g.set_entry_point("emotion")
    g.add_edge("emotion", "churn")
    g.add_edge("churn", "recommend")
    g.add_edge("recommend", "memory")
    g.set_finish_point("memory")

    return g.compile()
