import pandas as pd
import json
from prophet import Prophet


class Forecaster:
    def __init__(self):
        self.forecast = None
        self.model = Prophet()

    def fit(self, data):
        self.model.fit(data)

    def predict(self, forecast_period_days=365):
        # call after self.fit()
        future = self.model.make_future_dataframe(periods=forecast_period_days)
        forecast = self.model.predict(future)
        self.forecast = forecast
        return forecast

    def export(self, export_filename="forecast.xlsx"):
        # call after self.predict()
        # Выгружаем прогноз в эксель. Спрогнозированное значение лежит в столбце yhat
        self.forecast.to_excel(
            export_filename, sheet_name="Data", index=False, encoding="cp1251"
        )


def submit_price():
    price_forecaster = Forecaster()

    price_df = pd.read_csv("data.csv")
    price_df = price_df.rename(
        columns={"дата": "Date", "направление": "direction", "выход": "value"}
    )
    price_df = price_df[["Date", "value"]]

    price_df["Date"] = pd.to_datetime(price_df["Date"], dayfirst=True)
    price_df = price_df.iloc[::-1].reset_index(drop=True)

    price_df["value"] = price_df["value"].str.replace(",", ".").astype(float)

    price_df = price_df.rename(columns={"Date": "ds", "value": "y"})

    price_forecaster.fit(price_df)
    preds = price_forecaster.predict()

    # submit
    submit_df = pd.read_csv("test.csv")
    submit_df = submit_df.rename(
        columns={"дата": "Date", "направление": "direction", "выход": "value"}
    )
    submit_df = submit_df[["Date", "value"]]

    submit_df["Date"] = pd.to_datetime(submit_df["Date"], dayfirst=True)
    result_price = submit_df.merge(preds, left_on="Date", right_on="ds")[
        "yhat"
    ].tolist()

    with open("forecast_value.json", "w") as file:
        json.dump(result_price, file)


def submit_direction():
    direction_forecaster = Forecaster()

    direction_df = pd.read_csv("data.csv")
    direction_df = direction_df.rename(
        columns={"дата": "Date", "направление": "direction", "выход": "value"}
    )
    direction_df = direction_df[["Date", "direction"]]

    direction_df["Date"] = pd.to_datetime(direction_df["Date"], dayfirst=True)
    direction_df = direction_df.iloc[::-1].reset_index(drop=True)

    direction_df = direction_df.replace({"direction": {"ш": 0, "л": 1}})
    direction_df["direction"] = direction_df["direction"].astype(float)

    direction_df = direction_df.rename(columns={"Date": "ds", "direction": "y"})

    direction_forecaster.fit(direction_df)
    preds = direction_forecaster.predict()
    preds["yhat"] = preds["yhat"] > 0.5
    preds["yhat"] = preds["yhat"].astype(int)

    # submit
    submit_df = pd.read_csv("test.csv")
    submit_df = submit_df.rename(
        columns={"дата": "Date", "направление": "direction", "выход": "value"}
    )
    submit_df = submit_df[["Date", "direction"]]

    submit_df["Date"] = pd.to_datetime(submit_df["Date"], dayfirst=True)
    result = submit_df.merge(preds, left_on="Date", right_on="ds")["yhat"].tolist()

    with open("forecast_class.json", "w") as file:
        json.dump(result, file)


if __name__ == "__main__":
    submit_price()
    submit_direction()
