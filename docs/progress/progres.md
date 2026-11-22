# Stan pracy

| Etap        | Co robi                                                           | Stan                                                      |
|-------------|-------------------------------------------------------------------|-----------------------------------------------------------|
| `prepare`   | Pobranie HC3, filtr 50–150 słów, losowanie i sumy kontrolne       | Gotowy                                                    |
| `transform` | Synonimy WordNet, back-translation EN-DE-EN, parafraza T5         | Gotowy, z wznawianiem                                     |
| `filter`    | Stosunek długości, podobieństwo SBERT i ta sama decyzja detektora | Gotowy; próg SBERT do ustalenia po ręcznej ocenie         |
| `review`    | Próbka par do ręcznej oceny i podsumowanie progów SBERT           | Gotowy; ocena par do zrobienia                            |
| `explain`   | Atrybucje IG, LIME i SHAP (ferret), czas wyjaśnień                | Gotowy, z wznawianiem                                     |
| `metrics`   | Wyrównanie słów, Spearman, Top-k Jaccard, cosinus i baseline'y    | simalign, Spearman i Jaccard; cosinus i baseline'y tyg. 5 |
| `report`    | Tabele, wykresy i ranking metod XAI                               | Tabele per transformacja i metoda, W&B; wykresy tyg. 7    |

## Notatki tygodniowe

- [Tydzień 3](tydzien_03.md) - back-translation, parafraza, filtry jakości, ręczna ocena par
- [Tydzień 4](tydzien_04.md) - simalign, LIME i SHAP, pomiar czasu, W&B
