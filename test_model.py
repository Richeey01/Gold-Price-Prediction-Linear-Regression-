import joblib
import pandas as pd

model = joblib.load("gold_model.pkl")


while True:
    date_input = input("Enter date (YYYY-MM-DD): ")
    try:
        date = pd.to_datetime(date_input).toordinal()
        break
    except:
        print("Ошибка! Пожалуйста, введите дату заново. Пример: 2011-11-21")


while True:
    interest_input = input("Enter interest rate: ")
    try:
        interest_rate = float(interest_input)
        break
    except:
        print("Ошибка! Пожалуйста, введите число. Пример: 3.5")


while True:
    usd_input = input("Enter USD index: ")
    try:
        usd_index = float(usd_input)
        break
    except:
        print("Ошибка! Пожалуйста, введите число. Пример: 102.3")


X_new = pd.DataFrame(
    [[date, interest_rate, usd_index]],
    columns=["Date", "Interest_Rate", "USD_Index"]
)

# предсказание
prediction = model.predict(X_new)

print("Predicted gold price:", round(prediction[0], 2))