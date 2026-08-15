# 📊 Customer Churn Prediction Dashboard

An interactive **Machine Learning-powered Customer Churn Prediction Dashboard** that helps businesses identify customers who are likely to leave their service. The application combines **Data Analytics, Machine Learning, and interactive visualization** in a user-friendly Streamlit dashboard.

The project takes customer information, processes the data using a trained machine learning pipeline, predicts churn probability, and presents meaningful insights through an interactive dashboard.

---

## 🚀 Features

* 🤖 **Customer Churn Prediction**

  * Predict whether a customer is likely to churn.
  * Generate individual customer risk predictions.

* 📊 **Interactive Analytics**

  * Explore customer demographics and behavioral patterns.
  * Visualize important churn-related information.

* 🎯 **Risk Classification**

  * Classify customers according to their predicted churn risk.
  * Help identify customers who may require retention strategies.

* 📈 **Model Evaluation**

  * Accuracy
  * Precision
  * Recall
  * Confusion Matrix
  * ROC Curve / AUC

* 👥 **Customer Segmentation**

  * Analyze customer groups using clustering techniques.
  * Identify different customer behavior patterns.

* 🖥️ **Interactive Streamlit Dashboard**

  * Simple and responsive interface.
  * Real-time prediction based on customer inputs.

---

## 🛠️ Tech Stack

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Core programming language      |
| Pandas       | Data manipulation and analysis |
| NumPy        | Numerical operations           |
| Scikit-learn | Machine Learning               |
| Matplotlib   | Data visualization             |
| Joblib       | Model serialization            |
| Streamlit    | Interactive web dashboard      |

---

## 🧠 Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

```text
Customer Dataset
       ↓
Data Preprocessing
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Saved ML Model
       ↓
Streamlit Dashboard
       ↓
Customer Churn Prediction
```

---

## 📋 Customer Information

The model can use customer attributes such as:

* Gender
* Senior Citizen status
* Partner
* Dependents
* Tenure
* Contract type
* Payment method
* Internet service
* Monthly charges
* Total charges
* Other customer service information

These features are processed before being passed to the trained machine learning model.

---

## 📂 Project Structure

```text
customer-churn-dashboard/
│
├── data/
│   └── telco_churn.csv
│
├── models/
│   ├── churn_model.pkl
│   └── preprocessing.pkl
│
├── notebook/
│   └── exploration.py
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/customer_churn_prediction_dashboard.git
```

### 2. Navigate to the project directory

```bash
cd customer_churn_prediction_dashboard
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Dashboard

Start the Streamlit application using:

```bash
streamlit run app.py
```

The dashboard will open in your browser.

If it does not open automatically, Streamlit will provide a local URL in the terminal.

---

## 📊 Dashboard Capabilities

The dashboard provides an interactive environment where users can:

1. Enter customer information.
2. Submit the information to the prediction system.
3. Generate a churn prediction.
4. View the customer's estimated risk.
5. Analyze customer and churn patterns.
6. Review machine learning performance metrics.
7. Explore customer segments and insights.

---

## 🎯 Business Objective

Customer churn can significantly affect business revenue and long-term growth.

This project aims to help organizations move from a **reactive approach** to a **proactive retention strategy** by identifying customers who have a higher probability of leaving.

Instead of waiting until a customer leaves, businesses can use churn predictions to:

* Identify high-risk customers.
* Understand customer behavior.
* Prioritize retention campaigns.
* Improve customer experience.
* Reduce potential revenue loss.

---

## 🔮 Future Enhancements

Planned improvements include:

* Explainable AI using SHAP
* Advanced customer segmentation
* AI-powered retention recommendations
* Revenue-loss prediction
* Automated model retraining
* Real-time customer data integration
* Model comparison and optimization
* Cloud deployment
* AI chatbot for churn analysis
* Advanced business intelligence reports

---

## 🔐 Security

Sensitive information such as API keys, passwords, credentials, and environment variables should **never be committed to GitHub**.

Use a `.env` file for sensitive configuration and add it to `.gitignore`.

Example:

```text
.env
venv/
.venv/
__pycache__/
*.pyc
```

---

## 📚 Learning Outcomes

This project demonstrates practical experience with:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Machine Learning
* Model Evaluation
* Data Visualization
* Customer Segmentation
* Python Programming
* Streamlit Dashboard Development
* Git & GitHub

---

## 👨‍💻 Project

**Customer Churn Prediction Dashboard**

Built as a practical **Data Science and Machine Learning project** combining predictive analytics with an interactive business dashboard.

⭐ If you find this project useful, consider giving the repository a star!
