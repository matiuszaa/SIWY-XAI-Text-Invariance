# Design Proposal: Niezmienniczość wyjaśnień względem transformacji tekstu zachowujących znaczenie

**Kurs:** SIWY 2026/2027 (26Z)  
**Temat nr:** 3.2 (Witold Oleszkiewicz)  
**Zespół:** Mateusz Baran, Jakub Kowalczyk, Kinga Świderek
**Repozytorium:** https://github.com/matiuszaa/SIWY-XAI-Text-Invariance
**Data:** 24.10.2026

---

## 1. Cel projektu i pytanie badawcze

Głównym pytaniem badawczym jest: **Czy wyjaśnienia generowane dla detektorów tekstów tworzonych przez LLM pozostają
stabilne przy transformacjach zachowujących znaczenie tekstu, a jeśli nie, to które metody XAI są na nie najbardziej
odporne?**

Projekt ma wymiar eksperymentalny oraz narzędziowy: zbudujemy pipeline, który dokona semantycznie spójnych transformacji
tekstów, wygeneruje dla nich mapy atrybucji (wyjaśnienia) z wybranych detektorów, a następnie porówna te mapy w celu
oceny stabilności. Głównym produktem końcowym będzie udostępniony benchmark oraz ranking metod wyjaśnialności.

---

## 2. Planowane datasety

Skupimy się na tekstach w języku angielskim, ze względu na lepszą dostępność modeli detekcyjnych i narzędzi do
parafrazowania. Wybrano zbiory zróżnicowane pod kątem dziedziny:

| Dataset                                   | Typ danych                                        | Planowana próba                | Uwagi                                                                   |
|-------------------------------------------|---------------------------------------------------|--------------------------------|-------------------------------------------------------------------------|
| **HC3 (Human ChatGPT Comparison Corpus)** | Q&A, finanse, medycyna                            | 500 tekstów ludzkich / 500 LLM | Podstawowy zbiór referencyjny, bezpośrednie zestawienie odpowiedzi.     |
| **TweepFake**                             | Krótkie teksty (X/Twitter)                        | 1 000 wpisów                   | Sprawdzenie odporności dla krótkich, potocznych form.                   |
| **DAIGT (Detect AI Generated Text)**.     | Eseje uczniowskie, formalne teksty argumentacyjne | 500 tekstów ludzkich / 500 LLM | Sprawdzenie dla złożonej składni i parafraz. Wymaga przycięcia tekstów. |

---

## 3. Modele detekcyjne i metody wyjaśniania (XAI)

### 3.1 Modele detekcyjne (Black-box i White-box)

Do oceny wybierzemy 2-3 popularne detektory klasyfikujące tekst jako Human/AI:

- **RoBERTa-base OpenAI Detector** (klasyczny detektor oparty na transformerach).
- **DetectGPT / Binoculars** (detektory zero-shot oparte na prawdopodobieństwach i perplexity z modeli LLM).

### 3.2 Metody wyjaśnialności (Mapy atrybucji)

Wyjaśnienia będą przypisywać każdemu tokenowi w tekście wartość ("ważność" dla decyzji detektora):

- **Integrated Gradients (IG)** (wymaga dostępu do gradientów).
- **LIME (Local Interpretable Model-agnostic Explanations)** (model-agnostic, perturbacje na poziomie słów/tokenów).
- **SHAP (Token/Partition SHAP)**.

---

## 4. Transformacje tekstu i kontrola jakości (Meaning-Preservation)

Aby przetestować stabilność, oryginalne teksty zostaną poddane transformacjom. Musimy jednak zagwarantować, że znaczenie
i decyzja detektora pozostają niezmienne (zgodnie z założeniem projektu).

**Planowane transformacje:**

1. **Back-translation (Tłumaczenie w obie strony):** np. EN -> FR -> EN (używając modeli typu MarianMT).
2. **Parafraza (LLM-based):** Użycie modelu typu Llama 3 (prompt: "Rewrite the following text using different wording
   and syntax, but preserve the exact meaning, tone, and length. Do not add any new information.").
3. **Modyfikacje na poziomie słów (EDA - Easy Data Augmentation):** Synonimy z WordNet (zastąpienie 10-15% słów).

**Kontrola jakości (Filtr semantyczny):**
Każda para (oryginał, po transformacji) przejdzie przez model **Sentence-BERT (SBERT)**. Pary o podobieństwie
kosinusowym (Cosine Similarity) poniżej określonego progu (np. 0.85) oraz te, dla których **detektor zmienił decyzję (
tzw. label flip)**, zostaną odrzucone z analizy map atrybucji.

---

## 5. Zestaw metryk niezmienniczości (Invariance Metrics)

Ponieważ teksty po transformacji mają różną długość i układ tokenów, mapy atrybucji należy wyrównać (Token Alignment).
Użyjemy algorytmów typu Needleman-Wunsch, Dynamic Time Warping (DTW) lub prostej heurystyki opartej na embedingach słów.

**Ewaluacja stabilności (metryki):**

- **Rank Correlation (Spearman's ρ):** Korelacja rankingów atrybucji dla słów, które wystąpiły w obu tekstach.
- **Top-K Intersection (Jaccard Index):** Pokrycie się 10% najważniejszych tokenów (według metody XAI) między oryginałem
  a tekstem przekształconym.
- **Center of Mass Shift:** Przesunięcie środka masy atrybucji w obrębie zdania/tekstu.

---

## 6. Harmonogram

Harmonogram uwzględnia oficjalne terminy (deadline'y) dla semestru 26Z:

| Tydzień (Data)                           | Zakres prac                                                                                                                                                                                  | 
|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|  
| **Tydzień 1 (26.10 – 01.11.2026)**       | Złożenie dokumentu design proposal do 28.10.2026. Konfiguracja repozytorium (cookiecutter, venv, bazowe środowisko).                                                                         | 
| **Tydzień 2 (02.11 – 08.11.2026)**       | Deadline na prototyp - 06.11.2026. Przygotowanie działającego pipeline'u na małej próbce tekstów. Integracja 1 detektora, 1 metody (IG) oraz 1 transformacji (EDA). Spotkanie z prowadzącym. |
| **Tydzień 3 (09.11 – 15.11.2026)**       | Implementacja mechanizmów back-translation. Wdrożenie parafrazowania za pomocą modeli LLM.                                                                                                   |
| **Tydzień 4 (16.11 – 22.11.2026)**       | Wdrożenie filtra SBERT do weryfikacji wierności semantycznej przekształconych tekstów.                                                                                                       | 
| **Tydzień 5 (23.11 – 29.11.2026)**       | Ewaluacja i filtrowanie tekstów pod kątem zmiany decyzji detektora (label flip)po zastosowanych transformacjach.                                                                             |
| **Tydzień 6 (30.11 – 06.12.2026)**       | Rozpoczęcie implementacji metod LIME oraz SHAP przy użyciu biblioteki Captum/Transformers Interpret.                                                                                         |                                                                                                                                                                        
| **Tydzień 7 (07.12 – 13.12.2026)**       | Praca nad rozwiązaniem problemu wyrównywania tokenów (Token Alignment) pomiędzy tekstem oryginalnym a przetransformowanym.                                                                   |
| **Tydzień 8 (14.12 – 20.12.2026)**       | Zakończenie prac nad wyrównywaniem tokenów i finalizacja modułów XAI przed pełnym eksperymentem.                                                                                             |
| **Tydzień 9 (21.12 – 27.12.2026)**       | Odpalenie pełnego eksperymentu. Rozpoczęcie obliczania map atrybucji oraz metryk niezmienniczości.                                                                                           |
| **Tydzień 10 (28.12.2026 – 03.01.2027)** | Zakończenie obliczeń dla wszystkich metod i transformacji. ostatecznych wyników i stworzenie benchmarku.                                                                                     |
| **Tydzień 11 (04.01 – 10.01.2027)**      | Analiza zebranych wyników: tworzenie rankingu metod XAI pod kątem stabilności. Generowanie wykresów oraz przygotowanie rekomendacji końcowych.                                               |
| **Tydzień 12 (11.01 – 17.01.2027)**      | Termin zwolnienia (deadline: 14.01.2027).                                                                                                                                                    |

---

## 7. Planowana struktura projektu

```text
project/
├── data/
│   ├── raw/                  # Pobierane skryptem (HC3, TweepFake)
│   ├── transformed/          # Wyniki po back-translation/parafrazach
│   └── filtered/             # Dane po kontroli SBERT i checku detektora
├── src/
│   ├── transformations/      # Moduł z augmentacją, tłumaczeniem i parafrazą
│   ├── detectors/            # Wrappery na modele Hugging Face / API
│   ├── explainers/           # IG, LIME, SHAP zintegrowane z detektorami
│   ├── alignment/            # Logika wyrównywania tokenów (DTW/heurystyki)
│   └── metrics/              # Spearman, Top-K Jaccard
├── notebooks/                # Eksploracja danych i generowanie wykresów
├── tests/                    # Unit testy (pytest)
├── docs/                     # Analiza literaturowa, instrukcje
├── Makefile
├── requirements.txt          # Zależności
└── README.md
```

---

## 8. Stack technologiczny i zasoby obliczeniowe

**Technologie:**

- **Język:** Python 3.11
- **NLP & Transformacje:** Hugging Face `transformers`, `datasets`, `sentence-transformers` (SBERT), spacy/NLTK.
- **XAI:** `captum`, `shap`.
- **Eksperymenty:** Weights & Biases (W&B) do monitorowania metryk.
- **Standardy:** `pytest`, `black`, `ruff`.

**Zasoby obliczeniowe:**
Wymagany dostęp do GPU (1 stacja robocza/Google Colab) do uruchamiania LLM do parafrazowania, generowania map atrybucji
(zintegrowane gradienty wymagają wielokrotnego przejścia przez model) oraz osadzania tekstów przez SBERT. Całość
obliczeń powinna zamknąć się w kilku dniach.

---

## 9. Analiza literaturowa (szkic)
