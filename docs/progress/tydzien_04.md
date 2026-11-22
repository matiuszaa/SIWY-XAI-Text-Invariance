# Tydzień 4 (plan: 16.11 - 22.11, realizacja: 22.11)

## Co zostało zrobione

- `alignment.py`: wyrównanie słów simalign (mBERT `google-bert/bert-base-multilingual-cased`, metoda `itermax`).
  Dla każdego słowa oryginału zapisujemy listę słów transformacji, z którymi zostało wyrównane.
  Teksty dłuższe niż 510 subtokenów są odrzucane z błędem, bo simalign obcina je bez ostrzeżenia.
- `metrics` używa simalign dla back-translation i parafrazy, a dla synonimów mapy zapisanej w trakcie podmiany.
  Gdy słowo oryginału jest wyrównane z kilkoma słowami, ich atrybucje sumujemy (tak samo jak subtokeny w słowie).
  Słowa bez wyrównania pomijamy, a ich odsetek raportujemy jako `alignment_coverage`.
- LIME (1000 próbek, `seed` 42) i Partition SHAP (`max_evals` 500) przez ferret, obok IG.
- `explain` liczy wszystkie metody z sekcji `explainers` i każde wyjaśnienie zapisuje raz dla danego tekstu,
  a nie dla każdej pary (oryginał był liczony trzy razy). Dla LIME liczymy drugi bieg na oryginale
  z `rerun_seed` 43, który posłuży jako punkt odniesienia dla szumu metody.
- Czas każdego wyjaśnienia i urządzenie są zapisywane w `explanations.parquet`; `explain` wypisuje podsumowanie
  czasów, a raport ma osobną tabelę.
- `report` podaje wyniki per (transformacja, metoda XAI), odrzucenia per powód i liczbę zaakceptowanych par
  ludzkich i ChatGPT.
- Wyniki z `report` są logowane do W&B (`tracking.py`): konfiguracja, odsetek akceptacji, czasy i średnie metryki.

## Wybór metody wyrównania

Porównanie na 10 parach z `smoke` (odsetek słów oryginału, które dostały wyrównanie):

| Metoda | Back-translation | Parafraza |
|---|---|---|
| `inter` | 93 - 100 % | 67 - 85 % |
| `itermax` | 99 - 100 % | 78 - 95 % |
| `mwmf` | 96 - 100 % | 89 - 100 % |

`mwmf` wyrównuje prawie każde słowo, także takie, które w parafrazie nie ma odpowiednika, a `inter` gubi dużo
słów parafrazy. Wybraliśmy `itermax`, zalecany przez autorów simalign dla mBERT. Wyrównania nie są bezbłędne
(np. "American" -> "produced", "band" -> "Aerosmith."), co dodaje szum do metryk parafrazy.

## Pomiar czasu (CPU, prawdziwy detektor, jeden tekst 59 słów)

| Metoda | Ustawienia | Czas |
|---|---|---:|
| IG | 50 kroków | 42 s |
| LIME | 1000 próbek | 147 s |
| Partition SHAP | 500 ewaluacji | 121 s |

Na CPU pełny bieg nie jest realny (około 500 wyjaśnień na metodę). Decyzję o liczbie tekstów (100 czy 200)
podejmiemy po biegu `small.yaml` w Colabie, na podstawie tabeli czasów z raportu. Przy szacunkach z design
proposal (LIME 5 - 10 s, SHAP 5 - 15 s na T4) 100 tekstów to około 3 - 4 godziny GPU.

## Obserwacje

- **LIME jest bardzo zaszumiony.** Na tekście ChatGPT (p = 0.999) korelacja Spearmana między biegami z ziarnem 42
  i 7 wyniosła 0.14, przy identycznym tekście. Prawdopodobnie detektor jest nasycony: perturbacje prawie nie
  zmieniają prawdopodobieństwa, więc regresja LIME dopasowuje głównie szum. Jeśli to się potwierdzi, niska
  niezmienniczość LIME nie będzie wynikać z transformacji, tylko z samej metody. Dlatego potrzebny jest
  punkt odniesienia z drugiego biegu. Do omówienia z prowadzącym: czy liczyć LIME na logitach zamiast
  na prawdopodobieństwach.
- SHAP daje identyczny wynik niezależnie od ziarna, więc jego górna granica (ponowny bieg) wynosi 1, tak jak
  zakładał design proposal.
- Na `smoke` (losowy mini-detektor, wyniki tylko poglądowe) kolejność stabilności jest zgodna z oczekiwaniami:
  synonimy > back-translation > parafraza dla IG i SHAP, a LIME jest na poziomie szumu.

## Do zrobienia (tydzień 5)

- Bieg `small.yaml` w Colabie: ręczna ocena par, próg SBERT, tabela czasów i decyzja o liczbie tekstów.
- Podobieństwo kosinusowe, punkty odniesienia (losowa atrybucja, drugi bieg LIME) i bootstrap.
- Testy metryk na przypadkach brzegowych (np. identyczne teksty dają korelację 1).
