# ☕ Coffee Prediction AI

A fun and interactive machine learning project that predicts a person’s coffee choice based on their **mood**, **time of day**, **weather**, and **day of the week** using a Random Forest Classifier.

🔗 **Live App:** [https://coffee-prediction.streamlit.app/](https://coffee-prediction.streamlit.app/)  
📖 **Blog:** [Read on Medium](https://medium.com/@stutiagrawal61/how-i-used-ai-to-predict-my-friends-coffee-choice-%EF%B8%8F-and-why-it-worked-57d6ce251423)  
💻 **GitHub Repo:** [https://github.com/Stutii25/Coffee-prediction-sytem](https://github.com/Stutii25/Coffee-prediction-sytem)  
👤 **Author LinkedIn:** [Stuti Agrawal](https://www.linkedin.com/in/stuti-agrawal-48918b27b/)

## 🧠 Overview

This project showcases a beginner-friendly application of machine learning to predict coffee preferences. Built using Python and Streamlit, it accepts user inputs like mood, day, weather, and time — and returns a predicted coffee choice such as **Latte**, **Espresso**, or **Cappuccino**.

It also displays model confidence and useful visual insights.

---

## 📽️ Demo

🔗 **Live App:** [https://coffee-prediction.streamlit.app/](https://coffee-prediction.streamlit.app/)

Simply input your:
- Mood
- Day of the week
- Weather
- Time of day

...and see what coffee your AI barista suggests! ☕

---

## ✨ Features

- Simple and fun user interface (built with Streamlit)
- Decision Tree-based prediction with Random Forest
- Visualization of feature importance
- Confidence score display
- Clean and beginner-friendly codebase

---

## 📊 Dataset Schema

The dataset was manually created and consists of the following columns:

| Column Name    | Type     | Description                         |
|----------------|----------|-------------------------------------|
| `day`          | string   | Day of the week (e.g., Monday)      |
| `weather`      | string   | Weather condition (e.g., Sunny)     |
| `time_of_day`  | string   | Morning or Afternoon                |
| `mood`         | string   | User mood (e.g., Happy, Tired)      |
| `coffee_choice`| string   | Target label (e.g., Latte, Espresso)|

---

## 🛠️ Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/coffee-predictor.git
cd coffee-predictor
````

2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install the dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

To run the app locally:

```bash
streamlit run coffee_predictor_app.py
```

Then open the provided URL in your browser to interact with the app.

---

## 🧰 Technologies Used

* **Python** - core programming language
* **Pandas** - data handling
* **Scikit-learn** - ML model (Random Forest Classifier)
* **Streamlit** - web interface
* **Matplotlib / Seaborn** - optional for visualizations

---

## 📁 Folder Structure

```
coffee-predictor/
├── coffee_predictor_app.py       # Main Streamlit app
├── data/                         # (Optional) Folder for data storage
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview
```

---

## 👤 Author

**Stuti Agrawal**
Coffee Coder | ML Beginner | Passionate about AI Projects
📬 [stutiagrawal61@gmail.com](mailto:stutiagrawal61@gmail.com)

---


```

---

Would you like me to export this as a `.md` file or auto-generate your `requirements.txt` file too?
```
