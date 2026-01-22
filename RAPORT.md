# Raport - Przewidywanie Intencji Zakupowych

### Opis problemu
Projekt dotyczy klasyfikacji binarnej przewidującej, czy użytkownik dokona zakupu w sklepie internetowym na podstawie danych o jego zachowaniu na stronie.

### Cel projektu
Na podstawie danych o zachowaniu użytkownika (liczba odwiedzonych stron, czas spędzony na stronie, wskaźniki odrzuceń itp.) przewidzi, czy użytkownik dokona zakupu (kolumna `Revenue`).

### Opis zbioru danych
Zbiór danych `online_shoppers_intention.csv` zawiera informacje o sesjach użytkowników w sklepie internetowym.

- **Liczba obserwacji**: 12,330
- **Liczba zmiennych**: 18 (17 cech + 1 zmienna docelowa)

### Opis zmiennych

#### Zmienne numeryczne:
- `Administrative`, `Informational`, `ProductRelated` - liczba odwiedzonych stron danego typu
- `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` - czas spędzony na stronach danego typu
- `BounceRates` - wskaźnik odrzuceń
- `ExitRates` - wskaźnik wyjść
- `PageValues` - wartość stron
- `SpecialDay` - bliskość specjalnego dnia (np. święta)

#### Zmienne kategoryczne:
- `Month` - miesiąc sesji
- `OperatingSystems` - system operacyjny
- `Browser` - przeglądarka
- `Region` - region
- `TrafficType` - typ ruchu
- `VisitorType` - typ odwiedzającego (Returning_Visitor, New_Visitor, Other)
- `Weekend` - czy sesja odbyła się w weekend

#### Zmienna docelowa:
- `Revenue` - czy użytkownik dokonał zakupu (TRUE/FALSE)

## Preprocessing

### Wykryte problemy
- **Brakujące wartości**: W naszym przypadku nie znaleźliśmy żadnych brakujących wartości
- **Typy danych**: Wszystkie zmienne kategoryczne wymagały kodowania
- **Niezbalansowanie klas**: Procentowy rozkład: False    84.52  | True     15.47

### Zastosowane transformacje

1. **Obsługa brakujących wartości**: 
   - Strategia: drop/mean/median/mode
   - Uzasadnienie: Usuwamy, gdy braki stanowią margines danych np.3%, Średnia, gdy nie ma outliners, Mediana na numerach, gdy mamy wartości odstające, Dominanta dla danych kategorycznych 

2. **Kodowanie zmiennych kategorycznych**:
   - Metoda: Label Encoding
   - Zakodowane zmienne: Month, VisitorType, Weekend, OperatingSystems, Browser, Region, TrafficType
   - Uzasadnienie: Label Encoding został wybrany, aby zachować strukturę danych i uniknąć zbyt wielu kolumn (w przeciwieństwie do One-Hot Encoding)

3. **Normalizacja**:
   - Metoda: StandardScaler (standaryzacja do średniej=0, std=1)
   - Uzasadnienie: Normalizacja jest ważna dla algorytmów wrażliwych na skalę (np. SVM, Logistic Regression)

## Analiza Eksploracyjna (EDA)

### Statystyki opisowe
Zmienna,Jednostka,Średnia,Mediana (50%),Odch. std.,Max
Administrative,strony,"2,32","1,00","3,32","27,00"
Administrative_Duration,sekundy,"80,82","7,50","176,78","3398,75"
ProductRelated,strony,"31,73","18,00","44,48","705,00"
ProductRelated_Duration,sekundy,"1194,75","598,94","1913,67","63973,50"
PageValues,wartość,"5,89","0,00","18,57","361,76"
ExitRates,%,"0,04","0,03","0,05","0,20"
BounceRates,%,"0,02","0,00","0,05","0,20"

### Wnioski z wizualizacji
| Zmienna | Jednostka | Średnia | Mediana (50%) | Odch. std. | Max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Administrative** | strony | 2,32 | 1,00 | 3,32 | 27,00 |
| **Administrative_Duration** | sekundy | 80,82 | 7,50 | 176,78 | 3398,75 |
| **ProductRelated** | strony | 31,73 | 18,00 | 44,48 | 705,00 |
| **ProductRelated_Duration** | sekundy | 1194,75 | 598,94 | 1913,67 | 63973,50 |
| **PageValues** | wartość | 5,89 | 0,00 | 18,57 | 361,76 |
| **ExitRates** | % | 0,04 | 0,03 | 0,05 | 0,20 |
| **BounceRates** | % | 0,02 | 0,00 | 0,05 | 0,20 |

### Analiza korelacji
Najsilniejsze korelacje ze zmienną docelową `Revenue`:

PageValues (0,49) – najsilniejszy pozytywny wpływ na konwersję.

ExitRates (-0,21) – wysoki współczynnik wyjść znacząco obniża szansę na zakup.

ProductRelated (0,16) – większa liczba odwiedzonych stron produktów sprzyja sprzedaży.

BounceRates (-0,15) – wysoki współczynnik odrzuceń negatywnie wpływa na przychód.

ProductRelated_Duration (0,15) – dłuższy czas na stronach produktów zwiększa szansę na transakcję.

## 5. Feature Engineering

### Utworzone nowe cechy

1. **TotalPages** = Administrative + Informational + ProductRelated
   - Uzasadnienie: Łączna liczba odwiedzonych stron może być lepszym predyktorem niż poszczególne wartości

2. **TotalDuration** = Administrative_Duration + Informational_Duration + ProductRelated_Duration
   - Uzasadnienie: Całkowity czas spędzony na stronie jest ważnym wskaźnikiem zaangażowania

3. **AvgPageDuration** = TotalDuration / TotalPages
   - Uzasadnienie: Średni czas na stronę może wskazywać na jakość interakcji

4. **BounceExitRatio** = BounceRates / ExitRates
   - Uzasadnienie: Stosunek odrzuceń do wyjść może być wskaźnikiem jakości sesji

5. **ProductRelatedRatio** = ProductRelated / TotalPages
   - Uzasadnienie: Proporcja stron produktowych może być silnym predyktorem intencji zakupowej

### Selekcja zmiennych
- **Metoda**: Feature Importance (Random Forest)
- **Próg**: 0.01
- **Liczba wybranych cech**: 19
- **Uzasadnienie**: Random Forest pozwala na określenie ważności cech, co pomaga w usunięciu zmiennych o niskiej wartości predykcyjnej

## 6. Modelowanie

### Porównanie modeli

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | [0,8597 ] | [0,5354 ] | [0,7120 ] | [0,6112 ] | [0,8787 ] |
| Random Forest | [0,9015 ] | [0,7704 ] | [0,5183 ] | [0,6197 ] | [0,9118 ] |
| SVM | [0,8520 ] | [0,5156 ] | [0,7356 ] | [0,6063 ] | [0,8807 ] |

### Wybór najlepszego modelu
**Najlepszy model przed tuningiem**: Random Forest
- **Uzasadnienie**: Random Forest został wybrany jako najlepszy model, ponieważ osiągnął najwyższą dokładność (90,15%) oraz najlepszy wskaźnik ROC-AUC (0,9118). Wyróżnił się on najwyższą precyzją, co minimalizuje liczbę błędnych prognoz o potencjalnych zakupach. Algorytm ten najskuteczniej radzi sobie z nieliniowymi zależnościami w danych oraz specyficznym rozkładem kluczowej zmiennej PageValues.

### Optymalizacja hiperparametrów

**Model**: Random Forest

**Metoda**: Grid Search CV (5-fold)

**Hiperparametry testowane**:
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 20, 30, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

**Najlepsze parametry**: Random Forest
- `n_estimators`: [100]
- `max_depth`: [30]
- `min_samples_split`: [2]
- `min_samples_leaf`: [4]

**Uzasadnienie wyborów**: Grid Search pozwala na systematyczne przeszukanie przestrzeni parametrów, a cross-validation zapewnia niezawodną ocenę wydajności.

## 7. Wyniki i Ewaluacja

### Metryki najlepszego modelu (Random Forest - Tuned)

- **Accuracy**: 0.8925
- **Precision**: 0.6370
- **Recall**: 0.7120
- **F1-score**: 0.6724
- **ROC-AUC**: 0.9174

### Analiza Confusion Matrix

| | Przewidziano: No Purchase | Przewidziano: Purchase |
|--|-------------------------|----------------------|
| **Rzeczywistość: No Purchase** | TN: 1929 | FP: 155 |
| **Rzeczywistość: Purchase** | FN: 110 | TP: 272 |

**Interpretacja**:
- **True Positives (TP)**: 272 - Poprawnie przewidziane zakupy
- **False Positives (FP)**: 155 - Błędnie przewidziane zakupy (użytkownik nie kupił)
- **False Negatives (FN)**: 110 - Przeoczone zakupy (użytkownik kupił, ale model nie przewidział)
- **True Negatives (TN)**: 1929 - Poprawnie przewidziane braki zakupów

### Wnioski z ROC Curve
- **AUC Score**: 0.9174
- **Interpretacja**: Wynik powyżej 0.9 oznacza bardzo wysoką zdolność klasyfikatora do rozróżniania między użytkownikami dokonującymi zakupu a tymi, którzy tylko przeglądają stronę. Model radzi sobie znacznie lepiej niż klasyfikator losowy.

### Cross-Validation Results
- **Średni F1-score (5-fold CV)**: 0.6891 (+/- 0.0265)
- **Interpretacja**: Model wykazuje stabilność wyników na różnych podzbiorach danych. Nie zaobserwowano gwałtownych wahań metryki F1, co świadczy o dobrej generalizacji.

## 8. Konkluzja

### Podsumowanie wyników
Model Random Forest okazał się najbardziej skutecznym algorytmem spośród testowanych (Logistic Regression, SVM, RF). Po optymalizacji hiperparametrów osiągnął wysoką ogólną dokładność (89,25%) oraz bardzo solidny wynik AUC (0,9174). Model wykazuje dobry balans między precyzją (63,7%) a recall (71,2%), osiągając F1-score na poziomie 67,24%. W porównaniu do wersji przed tuningiem, model zoptymalizowany ma wyższy recall (71,2% vs 51,8%), co oznacza, że lepiej wykrywa użytkowników faktycznie dokonujących zakupów, kosztem nieco niższej precyzji.

### Analiza błędów
1. **False Positives**: 155 - Model przewiduje zakup, ale użytkownik nie kupił
   - **Możliwe przyczyny**: Użytkownicy spędzający bardzo dużo czasu na stronach produktów i mający wysoki PageValues, którzy w ostatniej chwili zrezygnowali z transakcji.

2. **False Negatives**: 110 - Model nie przewiduje zakupu, ale użytkownik kupił
   - **Możliwe przyczyny**: "Szybcy kupujący" – użytkownicy, którzy od razu przechodzą do finalizacji zamówienia bez długiego przeglądania stron pomocniczych czy administracyjnych.


### Ograniczenia modelu

1. **Jakość danych**: Brak informacji o historycznych zakupach użytkownika (użytkownik powracający vs nowy) ogranicza zdolność predykcyjną.
2. **Niezbalansowanie klas**: Klasa pozytywna to tylko ok. 15% zbioru, co sprawia, że model naturalnie skłania się ku przewidywaniu braku zakupu.
3. **Brak danych kontekstowych**: Brak informacji o źródle ruchu (konkretne kampanie reklamowe) czy demografii.
4. **Overfitting**: Nie stwierdzono krytycznego overfittingu dzięki zastosowaniu walidacji krzyżowej i optymalizacji parametrów modelu.

### Wnioski końcowe
Projekt zakończył się sukcesem w postaci budowy modelu zdolnego do identyfikacji intencji zakupowych z wysoką trafnością (AUC > 0.9). Kluczową zmienną okazała się PageValues, która ma najsilniejszy związek z końcowym sukcesem transakcyjnym. Model w obecnej formie może być wykorzystany do personalizacji treści na stronie w czasie rzeczywistym dla najbardziej obiecujących segmentów użytkowników.

---

**Data utworzenia**: 01.2026
**Autorzy**: Nikodem Boniecki, Adam Remfeld, Kacper Nowaczyk 
