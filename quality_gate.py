
MINIMUM_ACCURACY = 0.80


def check_model_quality(accuracy):
    print("Customer churn model accuracy:", round(accuracy * 100, 2), "%")
    print("Minimum required accuracy:", MINIMUM_ACCURACY * 100, "%")

    if accuracy >= MINIMUM_ACCURACY:
        print("QUALITY GATE PASSED")
        return True
    else:
        print("QUALITY GATE FAILED")
        return False


if __name__ == "__main__":
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    dataset = "data/customer_churn_raw.csv"
    model_file = "artifacts/customer_churn_model.joblib"

    if not __import__("os").path.exists(model_file):
        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )

    if not __import__("os").path.exists(dataset):
        raise FileNotFoundError(
            "Dataset not found. Run train_model.py first."
        )

    data = pd.read_csv(dataset)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    data = data.dropna()

    X = data.drop(columns=["Churn"])
    y = data["Churn"]
    X = pd.get_dummies(X, drop_first=False)

    saved = joblib.load(model_file)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    X = X.reindex(columns=feature_columns, fill_value=0)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y
    )

    predictions = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, predictions)

    if not check_model_quality(model_accuracy):
        raise SystemExit(1)
