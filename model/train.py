"""
Moduł model.train
Odpowiedzialny za trenowanie modelu Random Forest i zapisywanie go do pliku.
"""

import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from data.process_data import get_processed_data


def train_model(data_path, save_dir):
    """
    Trenuje model na podstawie danych z pliku CSV i zapisuje go w podanym katalogu.

    Args:
        data_path (str): Ścieżka do przetworzonego lub surowego pliku CSV.
        save_dir (str): Ścieżka do katalogu, gdzie ma zostać zapisany model.pkl.
    """

    # 1. Wczytanie danych
    try:
        df = get_processed_data(data_path)
    except FileNotFoundError as e:
        print(f"Błąd : {e}")
        return

    # 2. Walidacja czy kolumna pass_exam istnieje w wynikowym DataFrame
    if 'pass_exam' not in df.columns:
        print("Błąd: Brak kolumny 'pass_exam' w danych.")
        return

    # 3. Przygotowanie x i y
    y = df['pass_exam']
    x = df.drop(columns=['pass_exam'])

    # Zapisujemy nazwy cech dla aplikacji
    feature_names = x.columns.tolist()

    # 4. Podział na zbiór treningowy i testowy
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # 5. Trenowanie
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # 6. Ewaluacja
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("-----------------------------------")
    print(f"Model wytrenowany! Dokładność (Accuracy): {accuracy:.2f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred))
    print("-----------------------------------")

    # 7. Zapisywanie
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, 'model.pkl')

    # Zapisujemy słownik: model + lista kolumn
    artifacts = {
        "model": model,
        "features": feature_names
    }

    joblib.dump(artifacts, model_path)
    print(f"Model zapisano w: {model_path}")


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)

    INPUT_DATA = os.path.join(project_root, 'data', 'raw', 'student-mat.csv')
    OUTPUT_DIR = os.path.join(project_root, 'model', 'saved_models')

    print(f"Katalog projektu: {project_root}")
    print(f"Ścieżka danych: {INPUT_DATA}")
    train_model(INPUT_DATA, OUTPUT_DIR)
