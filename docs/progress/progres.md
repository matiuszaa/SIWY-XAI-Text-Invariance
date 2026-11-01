# Stan pracy

| Etap        | Co robi                                                           | Stan                                                      |
|-------------|-------------------------------------------------------------------|-----------------------------------------------------------|
| `prepare`   | Pobranie HC3, filtr 50–150 słów, losowanie i sumy kontrolne       | Gotowy                                                    |
| `transform` | Synonimy WordNet, back-translation EN-DE-EN, parafraza T5         | Do zrobienia                                              |
| `filter`    | Podobieństwo SBERT, stosunek długości i ta sama decyzja detektora | Do zrobienia                                              |
| `explain`   | Predykcje i atrybucje IG (Ferret)                                 | Gotowy; LIME/SHAP i pary po transformacji są do zrobienia |
| `metrics`   | Wyrównanie słów, Spearman, Top-k Jaccard, cosinus i baseline'y    | Do zrobienia                                              |
| `report`    | Tabele, wykresy i ranking metod XAI                               | Do zrobienia                                              |
