import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("GoldUp.csv")

df = df[["Date", "Gold_Price", "Interest_Rate", "USD_Index"]]

df["Date"] = pd.to_datetime(df["Date"]).map(pd.Timestamp.toordinal)

X = df[["Date", "Interest_Rate", "USD_Index"]]
y = df["Gold_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", round(r2_score(y_test, y_pred), 3))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_pred.min(), y_pred.max()]
)

plt.xlabel("Real Gold Price")
plt.ylabel("Predicted Gold Price")
plt.title("Gold Price Prediction")

plt.savefig("static/graph.png")
plt.close()

joblib.dump(model, "gold_model.pkl")