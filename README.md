# SIWY-XAI-Text-Invariance

Czy wyjaśnienia (IG, LIME, SHAP) detektorów tekstów LLM są stabilne względem transformacji tekstu zachowujących
znaczenie? Projekt 3.2, SIWY 2026/2027.
Opis: [docs/design_proposal_projektu_3_2.md](docs/design_proposal_projektu_3_2.md).

Pipeline bierze teksty ludzkie i ChatGPT z [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3), przekształca je
(synonimy, back-translation, parafraza), odrzuca pary, w których zmieniło się znaczenie lub decyzja detektora
[`Hello-SimpleAI/chatgpt-detector-roberta`](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta), liczy
atrybucje per słowo i porównuje je między oryginałem a transformacją.

## Instalacja

Wymaga [uv](https://docs.astral.sh/uv/) i Pythona 3.11

```bash
make install
```

Bez `make`: `uv sync`, potem `uv run pre-commit install`.

## Użycie

Punktem wejścia jest CLI `xai-inv` (`src/xai_inv/cli.py`):

Wybrać można z kilku konfiguracji w `configs/`.

```bash
xai-inv run --config configs/smoke.yaml       # cały pipeline
xai-inv transform --config configs/small.yaml # pojedynczy etap
xai-inv review --config configs/small.yaml    # próbka par do ręcznej oceny
xai-inv review-summary --config configs/small.yaml
```

| Etap             | Co robi                                                                      | Stan                                 |
|------------------|------------------------------------------------------------------------------|--------------------------------------|
| `prepare`        | pobranie HC3, filtr 50-150 słów, losowanie                                   | gotowy                               |
| `transform`      | synonimy WordNet, back-translation EN-DE-EN (MarianMT), parafraza T5         | gotowy                               |
| `filter`         | stosunek długości, podobieństwo SBERT, ta sama decyzja detektora             | gotowy; próg SBERT do ustalenia      |
| `review`         | próbka par do ręcznej oceny (bez wyników SBERT)                              | gotowy                               |
| `review-summary` | porównanie ręcznej oceny z progami SBERT                                     | gotowy                               |
| `explain`        | atrybucje IG, LIME i SHAP (ferret) zagregowane do słów, czas każdego wyjaśnienia | gotowy                           |
| `metrics`        | wyrównanie słów (simalign), Spearman, Top-k Jaccard                          | gotowy; cosinus i baseline'y w tyg. 5 |
| `report`         | tabele per transformacja i metoda XAI, czasy, log do W&B                     | gotowy; wykresy i ranking w tyg. 7   |

`review` i `review-summary` nie są częścią `run`, bo wymagają ręcznej oceny par.

`transform` i `explain` zapisują postęp co 20 wyników i przy ponownym uruchomieniu liczą tylko brakujące
albo te, dla których zmieniła się konfiguracja. Przerwaną sesję w Colabie wystarczy uruchomić ponownie.

### Colab i W&B

`small.yaml` i `full.yaml` wymagają GPU i logują wyniki do W&B (projekt `siwy-xai-text-invariance`).
Przed pierwszym biegiem trzeba się zalogować:

```bash
uv sync
uv run wandb login
uv run xai-inv run --config configs/small.yaml
```

W `smoke.yaml` W&B jest wyłączone (`wandb.mode: disabled`), więc ta konfiguracja działa bez konta i bez GPU.
Pierwsze uruchomienie pobiera modele z Hugging Face (około 3 GB).

### Konfiguracje

| Plik                 | Teksty | Detektor                                               | Urządzenie | Do czego                                          |
|----------------------|--------|--------------------------------------------------------|------------|---------------------------------------------------|
| `configs/smoke.yaml` | 5      | mały RoBERTa z losowymi wagami (`hf-internal-testing`) | CPU        | sprawdzenie, czy kod działa; wyniki bez znaczenia |
| `configs/small.yaml` | 100    | `Hello-SimpleAI/chatgpt-detector-roberta`              | CUDA       | właściwy eksperyment                              |
| `configs/full.yaml`  | 200    | `Hello-SimpleAI/chatgpt-detector-roberta`              | CUDA       | rozszerzony eksperyment                           |

`smoke.yaml` używa prawdziwych modeli transformacji, SBERT i simalign, ale mniejszej liczby kroków IG, próbek LIME
i ewaluacji SHAP. Wszystkie progi i parametry metod są w plikach konfiguracji.

### Opis tworzonych plików w katalogu `data`

```text
data/raw/hc3/all.jsonl                        pobrany HC3
data/interim[/<config>]/texts.parquet         wylosowane pytania i odpowiedzi: text_id, label (0 człowiek, 1 ChatGPT), source, question, text, n_words
data/interim[/<config>]/texts.json            opis dla texts.parquet
data/interim[/<config>]/pairs.parquet         pary (oryginał, transformacja): pair_id, transformation, transformed_text, word_alignment (tylko synonimy), transformation_status, transform_hash
data/interim[/<config>]/filtered_pairs.parquet pairs.parquet + predykcje detektora, length_ratio, semantic_similarity, filter_status, filter_reason
data/interim[/<config>]/review.csv            próbka par do ręcznej oceny (meaning_preserved: tak/nie, notes)
data/interim[/<config>]/review_summary.md     wynik ręcznej oceny dla progów SBERT
data/interim[/<config>]/explanations.parquet  jeden wiersz na (tekst, metoda, bieg): text_key, method, run (main/rerun), words, scores, seconds
data/interim[/<config>]/metrics.parquet       jeden wiersz na (para, metoda): word_alignment, alignment_coverage, spearman, top_k_jaccard
data/interim[/<config>]/report.md             raport z tabelami
```

`run = rerun` to drugi bieg LIME na oryginale z innym ziarnem (`rerun_seed`), potrzebny do oceny szumu metody.
IG i SHAP są deterministyczne, więc dla nich drugiego biegu nie liczymy.
