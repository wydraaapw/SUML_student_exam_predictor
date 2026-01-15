# Projekt SUML - Czy uczeń zda egzamin?
 Aplikacja przewidująca szansę zdania egzaminu na podstawie wybranych cech ucznia.

## Autorzy
- Paweł Wydrych (s25983)
- Mikołaj Toczko (s28901)
- Jakub Dominiczak (s27928)

## O projekcie
Uczeń, wprowadzając swoje dane (np. ocena z ćwiczeń, czas nauki, liczba nieobecności), może zobaczyć, jak jego zachowanie i zaangażowanie wpływają na prawdopodobieństwo zdania egzaminu.  Dzięki temu projekt może mieć zarówno wartość praktyczną jak i rozrywkową.
Projekt został udostępniony jako aplikacja webowa. 

### Jak sprawdzić szansę na zdanie egzaminu?

1. Wprowadź dane używając pól i suwaków.
2. Naciśnij przycisk **Oblicz szansę**.
3. Pod przyciskiem pojawi się wynik.

   <img width="710" height="85" alt="image" src="https://github.com/user-attachments/assets/d975e69a-aa32-412d-b161-dcc68ab50557" />


## Wykorzystane technologie

| Technologia | Zastosowanie w projekcie |
| :--- | :--- |
| **Python 3.11** | Główny język programowania logiki i modelu. |
| **Pandas** | Manipulacja danymi, czyszczenie i przygotowanie datasetu. |
| **Scikit-learn** | Implementacja algorytmu *Random Forest*, podział danych, metryki oceny. |
| **Streamlit** | Budowa interfejsu webowego. |
| **Joblib** | Serializacja modelu do pliku `.pkl` i jego wczytywanie. |
| **Docker** | Konteneryzacja aplikacji w celu zapewnienia przenoszalności. |
| **venv** | Środowiko wirtualne do izolacji projektu. |

# Uruchomienie

1. Sklonuj repozytorium
   ```bash
   git clone https://github.com/wydraaapw/SUML_student_exam_predictor.git
   cd SUML_student_exam_predictor
   ```
   
### Docker
Działa na każdym systemie z zainstalowanym środowiskiem Docker - https://www.docker.com/

1. Bedąc w folderze głównym projektu zbuduj obraz:
```bash
docker build -t exam-app .
```
2. Uruchom kontener
 ```bash
 docker run -p 8501:8501 exam-ap
 ```
Aplikacja będzie dostępna pod adresem URL - http://127.0.0.1:8501/


   

