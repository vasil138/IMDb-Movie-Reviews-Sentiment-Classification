import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """
    Оцінює якість класифікаційної моделі на тренувальній,
    валідаційній та, за наявності, тестовій вибірках.

    Параметри:
    ----------
    model : object
        Навчена модель машинного навчання з методом predict().
        Для розрахунку ROC-AUC модель повинна мати метод
        predict_proba() або decision_function().

    X_train : array-like
        Ознаки тренувальної вибірки.

    y_train : array-like
        Цільова змінна тренувальної вибірки.

    X_val : array-like
        Ознаки валідаційної вибірки.

    y_val : array-like
        Цільова змінна валідаційної вибірки.

    X_test : array-like, optional
        Ознаки тестової вибірки.

    y_test : array-like, optional
        Цільова змінна тестової вибірки.

    Повертає:
    ----------
    pandas.DataFrame
        Таблицю з метриками Accuracy, F1-score та ROC-AUC
        для кожної вибірки.
    """

    metrics = {}

    # Формуємо список вибірок для оцінки моделі
    datasets = [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
    ]

    # Додаємо тестову вибірку, якщо вона передана
    if X_test is not None and y_test is not None:
        datasets.append(("Test", X_test, y_test))

    for dataset_name, X, y in datasets:

        # Отримуємо передбачення класів
        y_pred = model.predict(X)

        # Отримуємо ймовірності позитивного класу для ROC-AUC
        # Для моделей без predict_proba використовуємо decision_function
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]
        else:
            y_score = model.decision_function(X)

        # Розрахунок основних метрик класифікації
        metrics[dataset_name] = {
            "Accuracy": round(accuracy_score(y, y_pred), 4),
            "F1-score": round(f1_score(y, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y, y_score), 4),
        }

    # Повертаємо результати у вигляді таблиці
    return pd.DataFrame(metrics).T