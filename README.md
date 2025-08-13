# ⚡ ElectroCare AI Assistant Chatbot

ElectroCare AI Assistant Chatbot is an **NLP-based customer service chatbot** designed to help customers get information about ElectroCare company's **services, policies, and steps to perform certain actions** — all through an easy-to-use chat interface.  
It supports **multilingual conversations** in **English**, **Italiano**, and **Tigrigna**, allowing customers to interact in the language they are most comfortable with.

---

## ✨ Features

- 🗨 **Real-time chat** with an AI-powered intent classifier.
- 🌍 **Multilingual support**: English, Italiano, and Tigrigna.
- 🔍 **Knowledge Base Search** for fallback answers when intent confidence is low.
- 📚 **Intent classification** using TF-IDF + Logistic Regression with custom threshold tuning.
- 📝 **Suggestions & quick replies** for related topics.

---

## 🛠 Tech Stack

- **Backend**: Python, Flask, scikit-learn, joblib
- **Frontend**: HTML, CSS, Vanilla JS
- **NLP**: TF-IDF, Logistic Regression, threshold tuning
- **Multilingual Translation**: deep-translator
- **Data**: Custom `intents.json` & `kb_articles.json` knowledge base

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yaekobB/electroCare-ai-assistant-chatbot-project.git
cd electroCare-ai-assistant-chatbot-project
```

### 2️⃣ Install Dependencies
Make sure you have **Python 3.10+** installed.
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
python app/app.py
```
The app will start on **http://localhost:5000**

---

## 🌐 Usage

1. Select your **preferred language** (English, Italiano, Tigrigna) from the dropdown in the UI.
2. Type your question — e.g.,  
   - `"How do I return a product?"`  
   - `"Come posso restituire un prodotto?"`  
   - `"ኣነ ምምላስ እንተደሊኩ እንታይ ክግበር ኣለኒ?"`
3. Receive responses **in your selected language**, along with:
   - Related knowledge base links.
   - Suggested related questions.

---

## 📸 Screenshots

**Main UI**  [View Main Chat UI](assets/main-UI.png)

**English Chat Example**  [View English Chat](assets/chat-english.png)

**Italiano Chat Example**  [View Italian Chat](assets/chat-italiano.png)

**Tigrigna Chat Example**  [View Tigrigna Chat](assets/chat-tigrinya.png)


---

## 📂 Project Structure
```
electroCare-ai-assistant-chatbot-project/
│
├── app/                        # Application source code
│   ├── app.py                   # Main Flask backend (entry point)
│   ├── answers.py               # Predefined responses per intent
│   ├── intent_classifier_optimized.py  # Intent classification training & evaluation
│   ├── kb_build.py              # Build script for knowledge base index
│   └── static/
│       └── index.html           # Frontend chat UI
│
├── data/                        # Data files
│   ├── intents_optimized.json   # Training data for intent classifier
│   ├── test_data_optimized.json # External evaluation dataset
│   ├── kb_articles.json         # Knowledge base articles
│   └── nav_index.json           # Navigation index for website sections
│
├── models/                      # Saved ML models & label encoders
│
├── assets/                      # Images for README
│   ├── main-UI.png               # Main chatbot UI screenshot
│   ├── chat-english.png          # Chat example in English
│   ├── chat-italiano.png         # Chat example in Italian
│   └── chat-tigrinya.png         # Chat example in Tigrinya
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore rules
```
---

## 📊 Model Training & Evaluation
The ElectroCare AI Assistant Chatbot was trained and evaluated using an optimized NLP pipeline:  

- **Internal validation accuracy**: ~0.75  
  *Performance on a held-out portion of the training dataset.*  
- **External test accuracy**: ~0.84  
  *Performance on an unseen evaluation dataset.*  

**Training Process:**  
- **Text Preprocessing & Cleaning** – Normalization, lowercasing, and punctuation removal for consistent text representation.  
- **Feature Extraction** – TF-IDF vectorization with unigram and bigram support to capture contextual meaning.  
- **Model Training** – Logistic Regression classifier with per-class threshold tuning to improve intent classification accuracy.  
- **Data Augmentation** – Inclusion of *hard negatives* to increase robustness against ambiguous or misleading queries.

---

## 🔮 Future Improvements
- Integrate more advanced translation APIs for improved multilingual accuracy.
- Add voice input/output support for hands-free interaction.
- Improve Tigrigna translation quality with a specialized model.
- Enhance intent classification using transformer-based embeddings.
- Expand knowledge base for more comprehensive answers.

## 📜 License
This project is licensed under the **MIT License** — you are free to use, modify, and distribute it.

---
