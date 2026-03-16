from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# загрузка модели
model = joblib.load("gold_model.pkl")

# загрузка датасета
df = pd.read_csv("GoldUp.csv")

# нужные колонки
df = df[["Date", "Gold_Price", "Interest_Rate", "USD_Index"]]

# перевод даты
df["Date"] = pd.to_datetime(df["Date"]).map(pd.Timestamp.toordinal)

X = df[["Date", "Interest_Rate", "USD_Index"]]
y = df["Gold_Price"]

# делим данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# предсказания
y_pred = model.predict(X_test)

# метрики
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# информация о датасете
total_rows = len(df)
total_columns = len(df.columns)

used_columns = ["Date", "Interest_Rate", "USD_Index"]

sample_data = df.head(10).to_html(classes="table", index=False)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        date = pd.to_datetime(request.form["date"]).toordinal()
        interest = float(request.form["interest"])
        usd = float(request.form["usd"])

        X_new = pd.DataFrame(
            [[date, interest, usd]],
            columns=used_columns
        )

        prediction = model.predict(X_new)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        total_rows=total_rows,
        total_columns=total_columns,
        used_columns=used_columns,
        sample_data=sample_data,
        mse=round(mse, 4),
        r2=round(r2, 4)
    )


app.run(debug=True)