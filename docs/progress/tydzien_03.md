# Tydzień 3 (plan: 09.11 - 15.11, realizacja: 21.11 - 22.11)

## Odchylenie od harmonogramu

Prace z tygodnia 3 zostały wykonane z opóźnieniem, w weekend 21-22.11, razem z początkiem tygodnia 4.
Zakres się nie zmienił. Ręczne sprawdzenie par i ustalenie progu SBERT przesuwamy na pierwszy bieg
`small.yaml` w Colabie, bo lokalnie (CPU, bez GPU) da się uruchomić tylko konfigurację `smoke`.

## Co zostało zrobione

- `transform` uruchamia wszystkie transformacje z sekcji `transforms` w konfiguracji:
  - `synonyms` - podmiana synonimów WordNet (bez zmian względem prototypu),
  - `backtranslation` - EN -> DE -> EN, MarianMT (`Helsinki-NLP/opus-mt-en-de`, `opus-mt-de-en`),
  - `paraphrase` - parafraza zdanie po zdaniu, `humarin/chatgpt_paraphraser_on_T5_base`.
- Back-translation i parafraza dzielą tekst na zdania (NLTK `punkt_tab`) i zachowują podział na akapity.
- Wyniki transformacji są zapisywane do `pairs.parquet` co 20 par. Ponowne uruchomienie liczy tylko
  brakujące pary albo te, dla których zmieniła się konfiguracja (kolumna `transform_hash`).
- `filter` sprawdza kolejno: odrzucenie transformacji, stosunek długości w słowach (0.7 - 1.3),
  podobieństwo SBERT (`all-mpnet-base-v2`, próg startowy 0.85) i decyzję detektora. Para dostaje
  powód pierwszego niespełnionego warunku (`filter_reason`), a podobieństwo, stosunek długości
  i predykcje detektora są zapisywane także dla par odrzuconych.
- `review` eksportuje próbkę par do ręcznej oceny (`review.csv`, domyślnie 30 par, po równo
  z każdej transformacji). Plik nie zawiera wyniku SBERT ani decyzji filtra, żeby nie sugerować
  oceny. `review-summary` dołącza wyniki SBERT po ocenie i liczy, ile dobrych par traci
  i ile złych przepuszcza każdy próg z `review.thresholds`.

## Zmiany względem design proposal

- **TextAttack nie jest używany.** Z `transformers` 5.x import `textattack` kończy się błędem
  (`flair` importuje usunięty `TransfoXLTokenizer`). Back-translation korzysta bezpośrednio
  z tych samych modeli MarianMT, których używa `BackTranslationAugmenter`. Synonimy już
  w prototypie były zaimplementowane bez TextAttack.
- **Seed dla każdej pary.** Ziarno losowania jest liczone z `seed` i `pair_id`, a nie raz na
  transformację. Dzięki temu wznowiony bieg daje te same pary co bieg od zera. Konkretne
  podmiany synonimów różnią się przez to od tych z biegu prototypu.
- Zaakceptowana para ma `filter_reason = passed` (wcześniej `same_detector_decision`).

## Ustawienia parafrazera

Porównaliśmy trzy ustawienia generowania na 6 zdaniach z próbki `smoke`:

| Ustawienie | Uwagi |
|---|---|
| beam search, 4 wiązki | zdanie o Flo zmieniło sens ("is a figment of Flo's imagination") |
| 4 wiązki, `repetition_penalty=1.5`, `no_repeat_ngram_size=3` | te same wyniki co wyżej |
| 5 wiązek, `repetition_penalty=10`, `no_repeat_ngram_size=2` | sens zachowany w 5/6 zdań; jedno zdanie ze zmianą ("usually not audible" zamiast "invisible") |

Wybraliśmy trzeci wariant, najbliższy ustawieniom z karty modelu (bez diverse beam search).
Błędy znaczenia mają wyłapywać filtr SBERT i ręczna ocena.

## Wyniki pośrednie (`smoke`, 5 tekstów, CPU)

Transformacje policzone na prawdziwych modelach, filtr uruchomiony z prawdziwym detektorem
`Hello-SimpleAI/chatgpt-detector-roberta` (w `smoke.yaml` jest losowy mini-detektor).

| Transformacja | Poprawne | Stosunek długości | Podobieństwo SBERT | Odrzucone |
|---|---:|---|---|---|
| synonyms | 5 / 5 | 1.00 | 0.806 - 1.000 | 1 (SBERT) |
| backtranslation | 5 / 5 | 0.88 - 1.04 | 0.928 - 0.991 | 1 (decyzja detektora) |
| paraphrase | 5 / 5 | 0.78 - 1.03 | 0.865 - 0.981 | 2 (decyzja detektora) |

Obserwacje:

- **Wszystkie zmiany decyzji detektora dotyczą tekstów ludzkich** po back-translation lub
  parafrazie: p(ChatGPT) rośnie z 0.000 do 0.90 - 0.998. Teksty ChatGPT i synonimy zachowały
  decyzję. Jeśli to się potwierdzi na `small`, filtr decyzji usunie większość ludzkich par dla tych
  transformacji i benchmark będzie niezbalansowany względem etykiety. Trzeba to raportować
  osobno dla każdej etykiety i omówić z prowadzącym.
- Teksty ludzkie z HC3 mają rozdzieloną interpunkcję ("current ." i "we 'll"), a back-translation
  i parafraza ją normalizują. To może być część powodu, dla którego detektor zmienia decyzję.
- Odrzucona podmiana synonimów: person -> mortal, just -> hardly, body -> soundbox. WordNet
  nie bierze pod uwagę części mowy ani znaczenia słowa. Filtr SBERT to wyłapał, ale warto
  rozważyć wybór synonimów z uwzględnieniem części mowy.
- Czasy na CPU: back-translation 7 - 16 s na tekst, parafraza około 10 s na tekst, wczytanie
  modeli około 40 s. Cały `transform` dla 5 tekstów trwał 167 s, a ponowne uruchomienie
  (wszystko z cache) 25 s, głównie na import bibliotek.

## Do zrobienia

- Bieg `prepare`, `transform` i `filter` dla `small.yaml` w Colabie.
- Ręczna ocena 30 par (`xai-inv review`), podsumowanie (`xai-inv review-summary`) i decyzja
  o progu SBERT. Uzupełniony `review.csv` i `review_summary.md` dodamy do `docs/progress/`.
- Sprawdzenie odsetka odrzuceń i balansu etykiet po filtrze na pełnej próbce.
