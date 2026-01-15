"""
Moduł data.process_data
Odpowiedzialny za wczytanie, czyszczenie, mapowanie oraz wybranie cech do modelu.
"""

import os
import pandas as pd
import numpy as np


def get_processed_data(filepath='data/raw/student-mat.csv'):
    """
    Wczytuje dane, czyści je, mapuje wartości tekstowe na liczbowe
    oraz wybiera kolumny do trenowania na podstawie analizy feature importance.

    Args:
        filepath (str): Ścieżka do pliku CSV. Domyślnie 'data/raw/student-mat.csv'.

    Returns:
        pd.DataFrame: Dataframe z wybranymi cechami gotowymi dla modelu.
    """

    # Sprawdzenie istnienia pliku
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nie znaleziono pliku: {filepath}")

    # Wczytanie danych
    df = pd.read_csv(filepath, sep=';')

    # 1. Stworzenie zmiennej target (pass_exam)
    # Target: 1 jeśli G3 >= 10, w przeciwnym razie 0
    df['pass_exam'] = np.where(df['G3'] >= 10, 1, 0)

    # 2. Zmiana nazwy G1 na exercise_grade
    df = df.rename(columns={'G1': 'exercise_grade'})

    # 3. Mapowanie zmiennych binarnych (yes/no)
    yes_no_mapping = {'yes': 1, 'no': 0}
    columns_yes_no = [
        'schoolsup', 'famsup', 'paid', 'activities',
        'nursery', 'higher', 'internet', 'romantic'
    ]

    for col in columns_yes_no:
        df[col] = df[col].map(yes_no_mapping)

    # 4. Mapowanie pozostałych zmiennych (Label Encoding)
    mappings = {
        'sex': {'F': 1, 'M': 0},
        'school': {'GP': 1, 'MS': 0},
        'address': {'U': 1, 'R': 0},
        'famsize': {'LE3': 1, 'GT3': 0},
        'Pstatus': {'T': 1, 'A': 0},
        'Mjob': {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4},
        'Fjob': {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4},
        'reason': {'home': 0, 'reputation': 1, 'course': 2, 'other': 3},
        'guardian': {'mother': 0, 'father': 1, 'other': 2}
    }

    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    # 5.Wybieramy tylko te kolumny, które mają wysoki Feature Importance
    # oraz są istotne dla interfejsu użytkownika

    columns_to_keep = [
        'pass_exam',       # Target
        'exercise_grade',  # Ocena z ćwiczeń
        'failures',        # ITNy
        'absences',        # Nieobecności
        'studytime',       # Czas nauki
        'goout',           # Wyjścia
        'age',             # Wiek
        'Walc',            # Alkohol weekend
        'Dalc',            # Alkohol tydzień
        'health',          # Zdrowie
        'freetime',        # Czas wolny
        'romantic',        # Związek
        'sex',             # Płeć
        'traveltime'       # Czas podróży
    ]

    # Usuwanie kolumn, zostają tylko kolumny z listy columns_to_keep
    final_columns = [c for c in columns_to_keep if c in df.columns]
    df = df[final_columns]

    # 6. Usunięcie ewentualnych braków danych
    df = df.dropna()

    return df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'raw', 'student-mat.csv')
    try:
        data = get_processed_data(csv_path)
        print(f"Kształt danych: {data.shape}")
        print("Pozostałe kolumny:", data.columns.tolist())
        print("Podgląd:")
        print(data.head())
    except Exception as e:
        print(f"Błąd: {e}")
