"""
Moduł app.main
Interfejs użytkownika wykonany przy wykorzystaniu iblioteki Streamlit.
"""

import os
import sys
import pandas as pd
import streamlit as st
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

st.set_page_config(page_title="Kalkulator Zdawalności", layout="centered")

def load_model():
    model_path = os.path.join(project_root, 'model', 'saved_models', 'model.pkl')
    if not os.path.exists(model_path):
        st.error("Brak modelu, uruchom: python -m model.train")
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Błąd modelu: {e}")
        st.stop()

def main():
    st.title("Czy zdasz egzamin?")
    st.markdown("Wprowadź swoje oceny i nawyki, aby sprawdzić szansę na sukces.")

    # 1. Wczytanie
    artifacts = load_model()
    model = artifacts["model"]
    feature_names = artifacts["features"]

    # 2. Słowniki
    grade_mapping = {
        1: 5,   # Niedostateczny
        2: 9,   # Dopuszczający
        3: 12,  # Dostateczny
        4: 15,  # Dobry
        5: 19   # Bardzo dobry
    }

    grade_desc = {
        1: '1 - Niedostateczny',
        2: '2 - Słaby / Dopuszczający',
        3: '3 - Dostateczny',
        4: '4 - Dobry',
        5: '5 - Bardzo dobry'
    }

    freq_opts = {1: 'Bardzo mało', 2: 'Mało', 3: 'Średnio', 4: 'Dużo', 5: 'Bardzo dużo'}
    health_opts = {1: 'Bardzo złe', 2: 'Złe', 3: 'Przeciętne', 4: 'Dobre', 5: 'Bardzo dobre'}
    study_opts = {1: '<2h', 2: '2-5h', 3: '5-10h', 4: '>10h'}
    travel_opts = {1: "<15 min", 2: "15-30 min", 3: "30 min - 1h", 4: ">1h"}
    mappings_sex = {'Kobieta': 1, 'Mężczyzna': 0}

    # 3. Formularz
    with st.form("exam_form"):
        st.subheader("Wyniki i Uczelnia")

        c1, c2 = st.columns(2)
        with c1:
            val_grade_ui = st.select_slider(
                "Ocena z ćwiczeń",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: grade_desc[x]
            )
            val_failures = st.number_input("Liczba ITNów", 0, 4, 0)

        with c2:
            val_absences = st.number_input("Liczba nieobecności w semestrze(suma godzin)",
                                           0, 93, 4,
                                           help="Wpisz orientacyjną liczbę opuszczonych godzin")
            val_study = st.select_slider("Czas nauki (tygodniowo)", options=[1, 2, 3, 4],
                                         format_func=lambda x: study_opts[x])

        st.divider()
        st.subheader("Styl życia i Zdrowie")

        col_health, col_alco = st.columns(2)

        with col_health:
            val_health = st.select_slider("Stan zdrowia", options=[1, 2, 3, 4, 5],
                                          value=5, format_func=lambda x: health_opts[x])
            val_travel = st.select_slider("Czas dojazdu", options=[1, 2, 3, 4],
                                          format_func=lambda x: travel_opts[x])

        with col_alco:
            val_dalc = st.select_slider("Alkohol (Dni robocze)", options=[1, 2, 3, 4, 5],
                                        value=1, format_func=lambda x: freq_opts[x])
            val_walc = st.select_slider("Alkohol (Weekend)", options=[1, 2, 3, 4, 5],
                                        value=2, format_func=lambda x: freq_opts[x])

        st.divider()
        st.subheader("Pozostałe informacje")

        c3, c4 = st.columns(2)
        with c3:
            sex_disp = st.radio("Płeć", list(mappings_sex.keys()))
            val_sex = mappings_sex[sex_disp]
            val_age = st.number_input("Wiek", 17, 30, 20)
        with c4:
            rom_disp = st.checkbox("W związku", value=False)
            val_romantic = 1 if rom_disp else 0
            val_goout = st.select_slider("Wyjścia ze znajomymi", options=[1, 2, 3, 4, 5],
                                         value=3, format_func=lambda x: freq_opts[x])
            val_freetime = st.select_slider("Czas wolny", options=[1, 2, 3, 4, 5], value=3,
                                            format_func=lambda x: freq_opts[x])


        submit = st.form_submit_button("Oblicz szansę")

    if submit:
        val_g1_mapped = grade_mapping[val_grade_ui]

        input_data = {
            'exercise_grade': val_g1_mapped,
            'failures': val_failures,
            'absences': val_absences,
            'studytime': val_study,
            'goout': val_goout,
            'age': val_age,
            'Walc': val_walc,
            'Dalc': val_dalc,
            'health': val_health,
            'freetime': val_freetime,
            'romantic': val_romantic,
            'sex': val_sex,
            'traveltime': val_travel
        }

        input_df = pd.DataFrame([input_data])

        try:
            # 2. Predykcja
            input_df = input_df[feature_names]
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]  # Pradwopodobieństwo zdania

            st.divider()

            if prediction == 1:
                st.balloons()
                st.success(f"###  PROGNOZA: ZDASZ! (Pewność: {probability:.1%})")
            else:
                st.error(f"###  PROGNOZA: RYZYKO NIEZDANIA (Szansa: {probability:.1%})")

        except KeyError as e:
            st.error(f"Błąd danych: {e}")

if __name__ == "__main__":
    main()