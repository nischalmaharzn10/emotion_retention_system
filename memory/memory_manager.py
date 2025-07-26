from typing import Optional, Dict, List, Any
from datetime import datetime
import uuid


class MemoryEntry:
    def __init__(self, user_input: str, ai_response: str, emotion_scores: Optional[Dict[str, float]] = None):
        """
        Represents a single conversational memory entry.

        Args:
            user_input (str): User's input message.
            ai_response (str): AI's response or recommendation.
            emotion_scores (Optional[Dict[str, float]]): Emotion scores associated with the input.
        """
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()
        self.user_input = user_input
        self.ai_response = ai_response
        self.emotion_scores = emotion_scores or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the memory entry to a dictionary format suitable for storage or JSON serialization.

        Returns:
            Dict[str, Any]: Serialized memory entry data.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "input": self.user_input,
            "response": self.ai_response,
            "emotion_scores": self.emotion_scores,
        }


class MemoryManager:
    def __init__(self):
        """
        Handles storing and retrieving conversation memory entries.
        """
        self._entries: List[MemoryEntry] = []

    def add(self, user_input: str, recommendation: str, emotion_scores: Optional[Dict[str, float]] = None) -> None:
        """
        Add a new memory entry to the conversation history.

        Args:
            user_input (str): User's input message.
            recommendation (str): AI's recommendation or response.
            emotion_scores (Optional[Dict[str, float]]): Emotion scores for the input.
        """
        entry = MemoryEntry(user_input, recommendation, emotion_scores)
        self._entries.append(entry)

    def get_context(self) -> List[Dict[str, str]]:
        """
        Get conversation history in a format compatible with LLM memory input.

        Returns:
            List[Dict[str, str]]: List of messages with 'role' and 'content', limited to last 20 entries.
        """
        history = []
        for entry in self._entries[-20:]:
            history.append({"role": "user", "content": entry.user_input})
            history.append({"role": "ai", "content": entry.ai_response})
        return history

    def get_full_memory(self) -> List[Dict[str, Any]]:
        """
        Retrieve full memory entries with metadata and emotion scores for last 20 interactions.

        Returns:
            List[Dict[str, Any]]: Serialized memory entries.
        """
        return [entry.to_dict() for entry in self._entries[-20:]]
