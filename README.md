# SIWY-XAI-Text-Invariance

Czy wyjaśnienia (IG, LIME, SHAP) detektorów tekstów LLM są stabilne względem transformacji tekstu zachowujących
znaczenie? Projekt 3.2, SIWY 2026/2027.
Opis: [docs/design_proposal_projektu_3_2.md](docs/design_proposal_projektu_3_2.md).

Pipeline bierze teksty ludzkie i ChatGPT z [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3), przekształca je
(synonimy, back-translation, parafraza), odrzuca pary, w których zmieniło się znaczenie lub decyzja detektora
[`Hello-SimpleAI/chatgpt-detector-roberta`](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta), liczy
atrybucje per słowo i porównuje je między oryginałem a transformacją.

## Instalacja

Wymaga [uv] i Pythona 3.11.

```bash
make install
```

Bez `make`: `uv sync`, potem `uv run pre-commit install`.

## Użycie

Punktem wejścia jest CLI `xai-inv` (`src/xai_inv/cli.py`):

Wybrać można z kilku konfiguracji w `configs/`.

```bash
xai-inv prepare --config configs/small.yaml
xai-inv explain --config configs/small.yaml
xai-inv run --config configs/smoke.yaml
```

| Etap        | Co robi                                                          | Stan                                                      |
|-------------|------------------------------------------------------------------|-----------------------------------------------------------|
| `prepare`   | pobranie HC3, filtr 50-150 słów, losowanie                       | gotowy                                                    |
| `transform` | synonimy WordNet, back-translation, parafraza T5                 | do zrobienia                                              |
| `filter`    | podobieństwo SBERT, stosunek długości, ta sama decyzja detektora | do zrobienia                                              |
| `explain`   | predykcje i atrybucje IG (ferret) zagregowane do słów oryginału  | gotowy; LIME/SHAP i pary po transformacji są do zrobienia |
| `metrics`   | wyrównanie słów, Spearman, Top-k Jaccard, cosinus, baseline'y    | do zrobienia                                              |
| `report`    | tabele, wykresy, ranking metod XAI                               | do zrobienia                                              |

`prepare` używa cache'u poprawnej próbki. `explain` zapisuje `explanations.parquet` od nowa; wznawianie
pozostałych etapów zostanie dodane razem z ich implementacją.

### Konfiguracje

| Plik                 | Teksty | Detektor                                               | Urządzenie | Do czego                                          |
|----------------------|--------|--------------------------------------------------------|------------|---------------------------------------------------|
| `configs/smoke.yaml` | 5      | mały RoBERTa z losowymi wagami (`hf-internal-testing`) | CPU        | sprawdzenie, czy kod działa; wyniki bez znaczenia |
| `configs/small.yaml` | 100    | `Hello-SimpleAI/chatgpt-detector-roberta`              | CUDA       | właściwy eksperyment                              |
| `configs/full.yaml`  | 200    | `Hello-SimpleAI/chatgpt-detector-roberta`              | CUDA       | rozszerzony eksperyment                           |

### Opis tworzonych plików w katalogu `data`

```text
data/raw/hc3/all.jsonl                        pobrany HC3
data/interim[/<config>]/texts.parquet         wylosowane pytania i odpowiedzi: text_id, label (0 człowiek, 1 ChatGPT), source, question, text, n_words
data/interim[/<config>]/texts.json            opis dla texts.parquet
data/interim[/<config>]/explanations.parquet  predykcja, słowa i wyniki IG dla każdego tekstu
```
