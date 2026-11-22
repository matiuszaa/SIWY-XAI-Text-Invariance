# Stan pracy

| Etap        | Co robi                                                           | Stan                                                      |
|-------------|-------------------------------------------------------------------|-----------------------------------------------------------|
| `prepare`   | Pobranie HC3, filtr 50–150 słów, losowanie i sumy kontrolne       | Gotowy                                                    |
| `transform` | Synonimy WordNet, back-translation EN-DE-EN, parafraza T5         | Gotowy, z wznawianiem                                     |
| `filter`    | Stosunek długości, podobieństwo SBERT i ta sama decyzja detektora | Gotowy; próg SBERT do ustalenia po ręcznej ocenie         |
| `review`    | Próbka par do ręcznej oceny i podsumowanie progów SBERT           | Gotowy; ocena par do zrobienia                            |
| `explain`   | Predykcje i atrybucje IG (Ferret)                                 | Gotowy; LIME/SHAP są do zrobienia                         |
| `metrics`   | Wyrównanie słów, Spearman, Top-k Jaccard, cosinus i baseline'y    | Spearman i Jaccard dla synonimów; simalign do zrobienia   |
| `report`    | Tabele, wykresy i ranking metod XAI                               | Raport z prototypu; podział na metody XAI do zrobienia    |

## Notatki tygodniowe

- [Tydzień 3](tydzien_03.md) - back-translation, parafraza, filtry jakości, ręczna ocena par
