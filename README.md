# ⚡ ElectroCare Assistant

ElectroCare Assistant is an **NLP-based customer service chatbot** designed to help customers get information about a company's **services, policies, and steps to perform certain actions** — all through an easy-to-use chat interface.  
It supports **multilingual conversations** in **English**, **Italiano**, and **Tigrigna**, allowing customers to interact in the language they are most comfortable with.

---

## ✨ Features

- 🗨 **Real-time chat** with an AI-powered intent classifier.
- 🌍 **Multilingual support**: English, Italiano, and Tigrigna.
- 🔍 **Knowledge Base Search** for fallback answers when intent confidence is low.
- 🖥 **Clean, responsive web UI** (optimized for desktop & mobile).
- 📚 **Intent classification** using TF-IDF + Logistic Regression with custom threshold tuning.
- 📝 **Suggestions & quick replies** for related topics.
- 🛡 **Debug mode** for developers to see intent and confidence scores.

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
git clone https://github.com/yourusername/electrocare-assistant.git
cd electrocare-assistant
```

### 2️⃣ Install Dependencies
Make sure you have **Python 3.10+** installed.
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
python app.py
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

**Main UI**  
![Main Chat UI](assets/ui_main.png)

**English Chat Example**  
![English Chat](assets/chat_english.png)

**Italiano Chat Example**  
![Italian Chat](assets/chat_italiano.png)

**Tigrigna Chat Example**  
![Tigrigna Chat](assets/chat_tigrigna.png)

---

## 📂 Project Structure
```
electrocare-assistant/
│── app.py                # Main Flask backend
│── index.html             # Frontend chat UI
│── intents.json           # Training data for intent classifier
│── test_data.json         # External evaluation dataset
│── kb_articles.json       # Knowledge base articles
│── nav_index.json         # Navigation index for website sections
│── models/                # Saved ML model files
│── assets/                # UI screenshots (for README)
│── requirements.txt       # Python dependencies
```
---

## 📊 Model Training & Evaluation
- **Internal validation accuracy**: ~0.75
- **External test accuracy**: ~0.84
- Training process:  
  - TF-IDF vectorization.  
  - Logistic Regression with per-class threshold tuning.  
  - Data augmentation with hard negatives.

---

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📜 License
This project is licensed under the **MIT License** — you are free to use, modify, and distribute it.

---

## 👨‍💻 Author
Developed by **[Your Name]**  
🎯 Master's Student in Artificial Intelligence & Computer Science  
📧 your.email@example.com
