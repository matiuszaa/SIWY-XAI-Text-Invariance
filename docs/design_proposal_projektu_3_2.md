# Design Proposal: Niezmienniczość wyjaśnień względem transformacji tekstu zachowujących znaczenie

**Kurs:** SIWY 2026/2027 (26Z)  

**Temat nr:** 3.2 (Witold Oleszkiewicz)  

**Zespół:** Mateusz Baran, Jakub Kowalczyk, Kinga Świderek

**Repozytorium:** https://github.com/matiuszaa/SIWY-XAI-Text-Invariance

**Data:** 24.10.2026

---

## 1. Cel projektu i pytanie badawcze

Pytanie badawcze: czy wyjaśnienia (mapy atrybucji) generowane dla detektorów tekstów tworzonych przez LLM pozostają stabilne, gdy tekst zostanie przekształcony w sposób zachowujący znaczenie, a jeśli nie, to która z badanych metod XAI jest na takie zmiany najbardziej odporna?

Projekt ma charakter eksperymentalny. Zbudujemy prosty pipeline, który:

1. bierze teksty ludzkie i wygenerowane przez ChatGPT ze zbioru HC3,
2. przekształca je trzema metodami zachowującymi znaczenie (podmiana synonimów, tłumaczenie w obie strony, parafraza),
3. odrzuca pary, w których znaczenie się zmieniło albo detektor zmienił decyzję,
4. liczy mapy atrybucji dla oryginału i transformacji trzema metodami XAI,
5. wyrównuje słowa między oryginałem a transformacją i liczy, jak bardzo mapy się różnią.

Wynikiem będzie mały benchmark (pary tekstów z metadanymi i policzonymi atrybucjami), zestaw metryk niezmienniczości, ranking metod XAI pod kątem stabilności oraz krótkie rekomendacje.

Spodziewamy się, że stabilność będzie spadać wraz z "inwazyjnością" transformacji (synonimy, potem back-translation, potem parafraza) oraz że LIME, jako metoda losowa, wypadnie najgorzej. Trzeba też pamiętać, że detektory tekstów LLM patrzą głównie na styl i dobór słów, a parafraza je zmienia, więc zmiana wyjaśnienia nie zawsze oznacza wadę metody XAI. Dlatego oprócz samego faktu zachowania decyzji będziemy zapisywać zmianę pewności detektora i porównywać wyniki z dwoma punktami odniesienia: losową atrybucją oraz ponownym uruchomieniem tej samej metody na tym samym tekście (LIME i SHAP mają własny szum).

---

## 2. Dane

Pracujemy na tekstach angielskich, bo dla nich są dostępne detektory i narzędzia. Ograniczamy się do jednego zbioru, żeby zmieścić się w czasie.

| Zbiór | Typ danych | Próba |
|---|---|---|
| HC3 (Guo i in., 2023) | odpowiedzi na pytania (Reddit ELI5, finanse, medycyna, Wikipedia), człowiek vs ChatGPT | 100 tekstów (50 ludzkich, 50 ChatGPT), długość 50-150 słów |

Ograniczenie długości do 50-150 słów wynika z limitu 512 tokenów w RoBERTa, kosztu LIME/SHAP i limitu 128 tokenów parafrazera (parafrazujemy zdanie po zdaniu). Liczba tekstów jest parametrem konfiguracji; jeśli obliczenia pójdą szybko, zwiększymy ją do 200. Rozważaliśmy też TweepFake i DAIGT, ale tweety są za krótkie, żeby parafraza zachowała sens, a DAIGT wymaga konta na Kaggle i przycinania tekstów.

---

## 3. Detektor i metody wyjaśniania

### 3.1 Detektor

Używamy klasyfikatora `Hello-SimpleAI/chatgpt-detector-roberta` (RoBERTa-base dotrenowana na HC3, od autorów zbioru). Jest mały, szybki i daje dostęp do gradientów, więc działają na nim wszystkie trzy metody XAI.

Nie bierzemy detektorów zero-shot (DetectGPT, Binoculars). DetectGPT liczy około 100 perturbacji T5 na jeden wynik, więc LIME z kilkuset próbkami to dziesiątki tysięcy przejść przez model na jedno wyjaśnienie, a Binoculars potrzebuje dwóch modeli Falcon-7B (około 28 GB VRAM). Nie zmieścimy tego w Colabie.

Jeśli zostanie czas, dołożymy drugi detektor, `openai-community/roberta-base-openai-detector` (trenowany na GPT-2), jako porównanie na modelu spoza rozkładu danych.

### 3.2 Metody XAI

Wszystkie trzy metody obsługuje biblioteka `ferret-xai`, która działa bezpośrednio na modelach z Hugging Face i sama agreguje atrybucje z subtokenów do słów:

- Integrated Gradients (50 kroków),
- LIME (1000 próbek, stały seed; drugi bieg z innym seedem służy do zmierzenia szumu metody),
- Partition SHAP (deterministyczny).

W razie problemów z `ferret` wracamy do Captum.

---

## 4. Transformacje tekstu i kontrola jakości

Korzystamy z gotowych implementacji, nie piszemy własnych.

| Transformacja | Narzędzie | Jak wyrównujemy słowa |
|---|---|---|
| Podmiana 10-15 % słów synonimami z WordNet (bez usuwania i przestawiania słów, bo to zmienia znaczenie) | `textattack.augmentation.WordNetAugmenter` | mapę zapisujemy w trakcie podmiany |
| Back-translation EN -> DE -> EN | `textattack.augmentation.BackTranslationAugmenter` (MarianMT) | `simalign` |
| Parafraza zdanie po zdaniu | `humarin/chatgpt_paraphraser_on_T5_base` (T5-base, 0.2B, bez gated access) | `simalign` |

Parafrazer T5 jest mały (0.2B) i nie wymaga akceptacji licencji ani dużej ilości VRAM, a wyniki i tak zapisujemy na dysku, więc benchmark nie zależy od użytego modelu.

Kontrola jakości pary (oryginał, transformacja):

1. podobieństwo kosinusowe SBERT (`all-mpnet-base-v2`) powyżej progu (startowo 0.85, próg sprawdzimy ręcznie na około 30 parach),
2. stosunek długości w słowach między 0.7 a 1.3,
3. ta sama decyzja detektora; różnicę prawdopodobieństwa zapisujemy do późniejszej analizy.

Odrzucone pary też zapisujemy, żeby móc podać odsetek odrzuceń dla każdej transformacji.

---

## 5. Wyrównanie i metryki niezmienniczości

Dla podmiany synonimów mapa wyrównania powstaje w trakcie transformacji. Dla back-translation i parafrazy używamy `simalign` (wyrównanie słów na embeddingach BERT, bez treningu, `pip install simalign`). Odsetek słów oryginału, które udało się wyrównać, raportujemy osobno.

Metryki liczone na wektorach atrybucji wyrównanych słów:

- korelacja rang Spearmana,
- Top-k Jaccard (k = 10 % słów) dla najważniejszych słów,
- podobieństwo kosinusowe.

Dla każdej metryki podajemy dwa punkty odniesienia: wartość dla losowej atrybucji (dolna granica) i dla ponownego uruchomienia tej samej metody na oryginale (górna granica, dla IG i SHAP równa 1). Wyniki agregujemy per (transformacja, metoda XAI) z przedziałami ufności z bootstrapu, i z tego powstaje ranking. Nie używamy metryki "Center of Mass Shift", bo pochodzi z prac o obrazach i przy parafrazie z przestawionymi zdaniami traci sens.

---

## 6. Planowana funkcjonalność programu

Jedno CLI (`typer`) z konfiguracją w plikach YAML (`configs/`), osobno od kodu:

```
xai-inv prepare     # pobranie HC3, filtr długości, próbkowanie, sumy kontrolne
xai-inv transform   # trzy transformacje + mapy wyrównania
xai-inv filter      # SBERT, długość, decyzja detektora
xai-inv explain     # atrybucje IG / LIME / SHAP dla każdej wersji tekstu
xai-inv metrics     # wyrównanie, metryki, baseline'y, bootstrap
xai-inv report      # tabele, wykresy, ranking
xai-inv run --config configs/small.yaml   # cały pipeline
```

Każdy etap zapisuje wynik do parquet i pomija teksty, które już policzył, więc przerwaną sesję w Colabie da się wznowić. Konfiguracja `smoke` (5 tekstów, mały losowy model RoBERTa z `hf-internal-testing`, CPU) służy do szybkiego sprawdzenia, czy pipeline działa.

Benchmark publikujemy jako release na GitHubie: `pairs.parquet` (oryginał, transformacja, metryki jakości, decyzje detektora), `explanations.parquet` (atrybucje per słowo), `results.parquet` i krótki opis.

---

## 7. Harmonogram

Terminy: design proposal 28.10.2026, prototyp 06.11.2026, oddanie 14.01.2027, prezentacja na ostatnim wykładzie.

| Tydzień (daty) | Zakres prac |
|---|---|
| 1 (26.10 - 01.11) | Oddanie design proposal (28.10) i wgranie go do repo. Szkielet repozytorium: cookiecutter-data-science, `uv` i `pyproject.toml`, ruff, pytest, pre-commit. Pobranie HC3, wrapper detektora, pierwszy test `ferret`. Start tabeli literatury w `docs/literature.md`. |
| 2 (02.11 - 08.11) | Prototyp (06.11): pipeline od początku do końca na 20 tekstach: synonimy -> detektor -> IG -> wyrównanie -> Spearman i Jaccard, wyniki w W&B. Uzupełniona tabela literatury. Spotkanie z prowadzącym. |
| 3 (09.11 - 15.11) | Back-translation i parafraza T5, zapis wyników na dysku. Filtr SBERT, długości i decyzji detektora. Ręczne sprawdzenie około 30 par i ustawienie progu. |
| 4 (16.11 - 22.11) | Wyrównanie `simalign`, agregacja tokenów do słów. LIME i SHAP przez `ferret`. Pomiar czasu na jedno wyjaśnienie i decyzja o liczbie tekstów. |
| 5 (23.11 - 29.11) | Komplet metryk z baseline'ami i bootstrapem. Testy jednostkowe metryk i wyrównania (np. identyczne teksty dają korelację 1). Bieg pilotażowy na 30 tekstach. |
| 6 (30.11 - 06.12) | Pełny bieg eksperymentu (100 tekstów, 3 transformacje, 3 metody XAI), nocne sesje w Colabie lub Kaggle. |
| 7 (07.12 - 13.12) | Analiza wyników: ranking metod, wykresy, zależność od zmiany pewności detektora. Bufor na powtórki, jeśli coś się wysypało. |
| 8 (14.12 - 20.12) | Szkic artykułu (struktura, related work z tabeli literatury, opis metody). Dokumentacja i instrukcja użytkowania w README. Jeśli jest czas: drugi detektor. |
| 9 (21.12 - 27.12) | Przerwa świąteczna, nie planujemy prac. |
| 10 (28.12 - 03.01) | Ograniczony zakres ze względu na przerwę świąteczną: drobne poprawki w tekście artykułu, porządki w repozytorium, przygotowanie plików benchmarku do publikacji. |
| 11 (04.01 - 10.01) | Filmik 3-5 min, prezentacja, końcowa wersja artykułu z rekomendacjami, release benchmarku na GitHubie. |
| 12 (11.01 - 17.01) | Oddanie projektu (14.01). Prezentacja na ostatnim wykładzie. |

---

## 8. Struktura projektu

```text
SIWY-XAI-Text-Invariance/
├── configs/                # YAML: smoke.yaml, small.yaml, full.yaml
├── data/
│   ├── raw/                # pobrane skryptem, niemodyfikowane, sumy kontrolne
│   ├── interim/            # transformacje, wyniki filtra, wyrównania
│   └── processed/          # explanations.parquet, results.parquet
├── src/xai_inv/
│   ├── cli.py
│   ├── data.py             # HC3
│   ├── transforms.py       # TextAttack, T5, mapy wyrównania
│   ├── quality.py          # SBERT, długość, decyzja detektora
│   ├── explainers.py       # ferret
│   ├── alignment.py        # simalign
│   ├── metrics.py          # Spearman, Jaccard, cosinus, baseline'y, bootstrap
│   └── report.py
├── notebooks/              # eksploracja, wykresy; funkcje przenosimy do src/
├── tests/
├── docs/                   # design proposal, literature.md, progress/, instrukcja
├── paper/
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 9. Stack technologiczny i zasoby obliczeniowe

Technologie: Python 3.11, `uv`, `transformers`, `datasets`, `sentence-transformers`, `textattack`, `simalign`, `ferret-xai` (zapasowo `captum`), `nltk` (podział na zdania), `typer`, YAML (`omegaconf`), `wandb`, `pandas`/`pyarrow`, `scipy`, `pytest`, `ruff`

Zasoby obliczeniowe (szacunek dla T4 w Colabie, RoBERTa-base, teksty około 200 tokenów):

| Komponent | Koszt na jednostkę | Wolumen | Razem |
|---|---|---|---|
| transformacje (WordNet, MarianMT, T5-base) | poniżej 0.5 s / tekst | 100 x 3 | kilka minut |
| filtr SBERT | ms / para | 300 par | około minuty |
| wyrównanie simalign | około 0.1 s / para | 300 par | około minuty |
| Integrated Gradients | około 0.5 s | 500 wyjaśnień | kilka minut |
| LIME (1000 próbek) | 5-10 s | 500 wyjaśnień | 1-1.5 h |
| Partition SHAP | 5-15 s | 500 wyjaśnień | 1-2 h |

500 wyjaśnień = 100 tekstów x 5 wersji (oryginał, 3 transformacje, powtórka oryginału). Łącznie około 3-4 godziny GPU, bez trenowania modeli. Podwojenie liczby tekstów lub dodanie drugiego detektora podwaja koszt, co nadal mieści się w darmowym Colabie lub Kaggle (30 h GPU tygodniowo).

---

## 10. Analiza literaturowa

Poniżej pozycje, na których opieramy projekt. Pełna tabela (z komentarzami uzupełnianymi w trakcie semestru) będzie w `docs/literature.md`.

| Praca | Po co nam | Kod / modele | Metryki, zasoby |
|---|---|---|---|
| Alvarez-Melis, Jaakkola (2018). On the Robustness of Interpretability Methods. [arXiv:1806.08049](https://arxiv.org/abs/1806.08049) | definicja stabilności wyjaśnień (podobne wejścia powinny dawać podobne wyjaśnienia), z której wychodzi cały temat | kod: tak | lokalna stała Lipschitza; małe modele |
| Ghorbani, Abid, Zou (2019). Interpretation of Neural Networks is Fragile. AAAI. [arXiv:1710.10547](https://arxiv.org/abs/1710.10547) | pokazuje, że małe perturbacje wejścia zmieniają mapy atrybucji; stąd bierzemy Spearmana i Top-k | kod: tak | Spearman, Top-k intersection; obrazy, GPU |
| Sinha i in. (2021). Perturbing Inputs for Fragile Interpretations in Deep NLP. BlackboxNLP. [ACL Anthology](https://aclanthology.org/2021.blackboxnlp-1.33/) | to samo pytanie co nasze, ale dla klasyfikatorów sentymentu: podmiana poniżej 10 % słów zmienia IG i LIME o ponad 20 % | kod: do sprawdzenia | korelacja rang, Top-k; BERT/RoBERTa |
| Schoenegger, Xia, Roth (2024). An Evaluation of Explanation Methods for Black-Box Detectors of Machine-Generated Text. [arXiv:2408.14252](https://arxiv.org/abs/2408.14252) | ocena SHAP i LIME dla detektorów tekstu LLM na HC3, z tym samym detektorem co u nas; podają czasy liczenia wyjaśnień i wskazują SHAP jako najstabilniejszy | kod: do sprawdzenia | Krippendorff alfa; LIME około 30 s / wyjaśnienie dla RoBERTa |
| Guo i in. (2023). How Close is ChatGPT to Human Experts? (HC3). [arXiv:2301.07597](https://arxiv.org/abs/2301.07597) | nasz zbiór danych i detektor | zbiór i model na HF Hub | F1; RoBERTa-base |
| Krishna i in. (2023). Paraphrasing evades detectors of AI-generated text. NeurIPS. [arXiv:2303.13408](https://arxiv.org/abs/2303.13408) | pokazuje, że parafraza mocno psuje detektory; uzasadnia filtrowanie label flipów i zapisywanie zmiany pewności | kod i model (11B, dla nas za duży) | skuteczność detekcji; T5-XXL |
| Attanasio i in. (2023). ferret: a Framework for Benchmarking Explainers on Transformers. EACL demo. [ACL Anthology](https://aclanthology.org/2023.eacl-demo.29/) | biblioteka, przez którą liczymy IG, LIME i SHAP na modelu HF | `pip install ferret-xai` | AOPC; CPU/GPU |
| Morris i in. (2020). TextAttack. EMNLP demo. [ACL Anthology](https://aclanthology.org/2020.emnlp-demos.16/) | gotowe augmentery: synonimy WordNet i back-translation | `pip install textattack` | - |
| Jalili Sabet i in. (2020). SimAlign. Findings of EMNLP. [ACL Anthology](https://aclanthology.org/2020.findings-emnlp.147/) | wyrównanie słów między oryginałem a parafrazą bez treningu | `pip install simalign` | alignment F1; mBERT |
| Reimers, Gurevych (2019). Sentence-BERT. EMNLP. [ACL Anthology](https://aclanthology.org/D19-1410/) | filtr podobieństwa znaczenia | modele na HF Hub | STS |
| Sundararajan i in. (2017) IG, [arXiv:1703.01365](https://arxiv.org/abs/1703.01365); Ribeiro i in. (2016) LIME, [arXiv:1602.04938](https://arxiv.org/abs/1602.04938); Lundberg, Lee (2017) SHAP, [arXiv:1705.07874](https://arxiv.org/abs/1705.07874) | trzy badane metody XAI | Captum, `lime`, `shap` | - |

Z literatury wynika, że metryki stabilności nie trzeba wymyślać (Ghorbani, Sinha), że detektory zero-shot są za drogie do wyjaśniania metodami perturbacyjnymi (Schoenegger), oraz że parafraza i synonimy realnie zmieniają zachowanie detektorów (Krishna), więc trzeba kontrolować zmianę decyzji i pewności. Gotowe biblioteki (ferret, TextAttack, SimAlign) pozwalają ograniczyć własny kod do spięcia pipeline'u, filtrów i metryk.

