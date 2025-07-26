# 🧠 Emotion-Aware Retention System

**Emotion-Aware Retention System** is a modular LangGraph-based AI workflow that proactively analyzes user sentiment, predicts churn risk, and delivers personalized retention actions, all while maintaining conversational context. From real‑time emotion detection to churn prediction, recommendation generation, and memory management, this system enables data‑driven engagement strategies.

---

## ✨ Features

### 🔍 Emotion Detection
- Fine-tuned HuggingFace transformer (`j-hartmann/emotion-english-distilroberta-base`)
- Returns confidence scores for emotions: joy, anger, sadness, disgust, fear, surprise, neutral

### 📈 Churn Prediction (PCI)
- Combines current and historical emotion scores with exponential decay
- Outputs a churn risk score between 0.0 and 1.0

### 🤖 Recommendation Engine
- Threshold-based actions:  
  - **ESCALATE** (high risk)  
  - **DEFUSE** / **SUPPORT** (medium-high risk + negative emotions)  
  - **CHECKIN** (moderate risk)  
  - **ENGAGE** (low risk + positive emotion)  
  - **OBSERVE** (default/passive)

### 🗄️ Memory Management (CMM)
- Appends user input, AI response, emotion scores, and churn risk to memory
- Retrieves last 20 interactions for context in subsequent requests

### 🖥️ Streamlit UI
- Interactive interface for entering messages, viewing emotion bar charts, churn probabilities, and recommendations
- JSON logging of clean, filtered results (no full conversation history)

---

## 🛠️ Tech Stack

- **Workflow Orchestration:** LangGraph (`StateGraph`)
- **Emotion Analysis:** HuggingFace Transformers + PyTorch
- **Memory & Context:** LangChain Memory via `MemoryManager`
- **Churn & Recommendation Logic:** Custom Python modules
- **UI:** Streamlit
- **Data Storage:** JSON (`data/mock_data.json`)

---

## 🚀 Getting Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/nischalmaharzn10/emotion_retention_system.git
   cd emotion_retention_system

2. **Create & activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt

4. **Run the UI**

   ```bash
   streamlit run ui/app.py

5. **Interact & Test!**

- Enter sample messages

- Observe emotion chart, churn risk %, and recommendation

- Check data/mock_data.json for logged output

---

## 📌 Future Enhancements
- Fine-tune emotion model on domain-specific data

- Incorporate user metadata (session length, account age) for richer churn modeling

- Add real‑time sentiment trend graphs and dashboard

- Integrate with CRM or customer support platforms

- Expand recommendation actions with multi-step workflows
