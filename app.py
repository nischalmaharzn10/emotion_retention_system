# === Imports ===
import json
from pathlib import Path
from memory.memory_manager import MemoryManager
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain.schema.messages import AIMessage
from graph.workflow import build_workflow
from graph.workflow import build_workflow
from typing import Dict, Any
import logging
import copy

logger = logging.getLogger(__name__)

# === Initialize Workflow ===
graph = build_workflow()

# === Streamlit UI ===
st.title("🧠 Emotion-Aware Retention System")
user_input = st.text_area("User message:")
output_path = Path("data/mock_data.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

# === Process Input ===
if st.button("Analyze") and user_input.strip():
    try:
        raw_result = graph.invoke({"input": user_input})

        # Normalize result
        if isinstance(raw_result, AIMessage):
            content = raw_result.content
            if isinstance(content, str):
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    st.error("AIMessage content is not valid JSON.")
                    st.stop()
            elif isinstance(content, dict):
                result = content
            else:
                st.error(f"Unsupported content type: {type(content)}")
                st.stop()
        elif isinstance(raw_result, dict):
            result = raw_result
        else:
            st.error(f"Unsupported result type: {type(raw_result)}")
            st.stop()

        # === Emotion Analysis ===
        emotions = result.get("emotion_scores", {})

        if not emotions:
            st.warning("No emotions detected.")
        else:
            highest_emotion = max(emotions, key=emotions.get)
            highest_confidence = emotions[highest_emotion]

            col1, col2 = st.columns([2, 3])

            with col1:
                st.subheader("Input Text")
                st.write(user_input)

                st.subheader("Detected Emotion")
                churn_risk = result.get("churn_risk")
                recommendation = result.get("recommendation")

                if churn_risk is not None and recommendation is not None:
                    st.subheader("Churn Risk")
                    st.markdown(f"**{churn_risk:.2%}**")

                st.subheader("Recommendation")

                # If recommendation is dict with 'message', display that; else fallback to string
                message = recommendation.get("message") if isinstance(recommendation, dict) else str(recommendation)

                st.markdown(message)

                st.markdown(f"**{highest_emotion.capitalize()}** with confidence **{highest_confidence:.2%}**")

            with col2:
                st.subheader("Emotion Scores")

                # DataFrame for Plot
                df = pd.DataFrame({
                    "Emotion": list(emotions.keys()),
                    "Confidence": list(emotions.values())
                }).sort_values("Confidence", ascending=True)

                # Horizontal Bar Chart
                fig = px.bar(
                    df,
                    x="Confidence",
                    y="Emotion",
                    orientation='h',
                    labels={"Confidence": "Confidence", "Emotion": "Emotion"},
                    text=df["Confidence"].apply(lambda x: f"{x:.1%}"),
                    width=600,
                    height=400,
                )

                fig.update_layout(
                    margin=dict(l=100, r=20, t=30, b=40),
                    yaxis=dict(tickfont=dict(size=14)),
                    xaxis=dict(range=[0, 1], fixedrange=True),
                    dragmode="zoom"
                )

                fig.update_traces(
                    marker_color='lightskyblue',
                    marker_line_color='blue',
                    marker_line_width=1.5
                )

                st.plotly_chart(fig, use_container_width=True)

        # === Save to JSON File ===
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                data = []
        else:
            data = []


        # Save result without history to avoid redundancy
        if isinstance(result, dict):
            result_to_save: Dict[str, Any] = result.copy()
            result_to_save.pop("history", None)  
            data.append(result_to_save)
        else:
            # Handle other types or ignore
            pass

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        # st.success(f"Analysis complete! Results saved to {output_path}")

    except Exception as e:
        st.error(f"Error during analysis or saving: {e}")
