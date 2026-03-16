# Gold Price Prediction

This project uses Linear Regression to predict the price of gold based on:

- Date
- Interest Rate
- USD Index

The model is trained using historical data and deployed in a simple web application.

---

# Project Structure

introToAI/

train_model.py – trains the model  
train_test.py – terminal prediction  
app.py – web application  
GoldUp.csv – dataset  
gold_model.pkl – trained model  

templates/index.html – website page  
static/graph.png – model graph  

---

# Requirements

Install the following libraries:

pip install pandas  
pip install scikit-learn  
pip install flask  
pip install matplotlib  
pip install joblib  

---

# How to Run the Project

Step 1. Clone the repository

git clone https://github.com/Richeey01/Gold-Price-Prediction-Linear-Regression-.git


Step 2. Go to project folder

cd introToAI


Step 3. Install dependencies

pip install pandas scikit-learn flask matplotlib joblib


Step 4. Train the model

python train_model.py


Step 5. Run the website

python app.py


Step 6. Open browser

http://127.0.0.1:5000



---

# Model Metrics

The model is evaluated using:

- R² score
- Mean Squared Error (MSE)

---

# Features

- Linear Regression model
- Dataset preview
- Prediction form
- Model accuracy display
- Graph visualization


