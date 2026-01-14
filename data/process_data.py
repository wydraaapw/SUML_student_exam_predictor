"""
Moduł data.process_data
Odpowiedzialny za wczytanie, czyszczenie i przygotowanie danych do procesu trenowania.
"""

import os
import pandas as pd
import numpy as np


def get_processed_data(filepath='data/raw/student-mat.csv'):
    """
        Wczytuje  dane, czyści je i mapuje wartości tekstowe na liczbowe
        aby były gotowe do trenowania.

        Args:
            filepath (str): Ścieżka do pliku CSV. Domyślnie 'data/raw/student-mat.csv'.

        Returns:
            pd.DataFrame: Dataframe z danymi gotowymi dla modelu.
        """

    # Sprawdzenie istnienia pliku
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nie znaleziono pliku: {filepath}")

    # Wczytanie danych
    df = pd.read_csv(filepath, sep=';')

    # 1. Stworzenie zmiennej target (pass_exam)
    # Target: 1 jeśli G3 >= 10, w przeciwnym razie 0, przedział = [0,20]
    df['pass_exam'] = np.where(df['G3'] >= 10, 1, 0)

    # 2. Usunięcie zbędnych kolumn
    # G3 - target, usuwamy
    # G2 - pomijamy
    cols_to_drop = ['G3', 'G2']

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 3. Zmiana nazwy G1 na exercise_grade
    df = df.rename(columns={'G1': 'exercise_grade'})

    # 4. Mapowanie zmiennych binarnych (yes/no)
    yes_no_mapping = {'yes': 1, 'no': 0}
    columns_yes_no = [
        'schoolsup', 'famsup', 'paid', 'activities',
        'nursery', 'higher', 'internet', 'romantic'
    ]

    for col in columns_yes_no:
        df[col] = df[col].map(yes_no_mapping)

    # 5. Mapowanie pozostałych zmiennych
    # Użycie mapowania liczbowego (Label Encoding)

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

    # 6. Usunięcie ewentualnych braków danych
    df = df.dropna()

    return df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'raw', 'student-mat.csv')
    try:
        data = get_processed_data(csv_path)
        print("Sukces! Dane przetworzone.")
        print(f"Kształt danych: {data.shape}")
        print("Podgląd:")
        print(data.head())
    except Exception as e:
        print(f"Błąd: {e}")
