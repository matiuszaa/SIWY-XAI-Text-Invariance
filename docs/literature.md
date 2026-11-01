# Literatura

Tabela startowa z design proposala (sekcja 10), uzupełniana w trakcie semestru.

| Praca | Po co nam | Kod / modele | Metryki, zasoby | Notatki |
|-------|-----------|--------------|-----------------|---------|
| Alvarez-Melis, Jaakkola (2018). On the Robustness of Interpretability Methods. [arXiv:1806.08049](https://arxiv.org/abs/1806.08049) | definicja stabilności wyjaśnień | kod: tak | lokalna stała Lipschitza | |
| Ghorbani, Abid, Zou (2019). Interpretation of Neural Networks is Fragile. AAAI. [arXiv:1710.10547](https://arxiv.org/abs/1710.10547) | Spearman i Top-k jako metryki stabilności | kod: tak | Spearman, Top-k intersection | |
| Sinha i in. (2021). Perturbing Inputs for Fragile Interpretations in Deep NLP. BlackboxNLP. [ACL Anthology](https://aclanthology.org/2021.blackboxnlp-1.33/) | to samo pytanie dla klasyfikatorów sentymentu | kod: do sprawdzenia | korelacja rang, Top-k | |
| Schoenegger, Xia, Roth (2024). An Evaluation of Explanation Methods for Black-Box Detectors of Machine-Generated Text. [arXiv:2408.14252](https://arxiv.org/abs/2408.14252) | SHAP i LIME dla detektorów tekstu LLM na HC3 | kod: do sprawdzenia | Krippendorff alfa | |
| Guo i in. (2023). How Close is ChatGPT to Human Experts? (HC3). [arXiv:2301.07597](https://arxiv.org/abs/2301.07597) | zbiór danych i detektor | HF Hub | F1 | |
| Krishna i in. (2023). Paraphrasing evades detectors of AI-generated text. NeurIPS. [arXiv:2303.13408](https://arxiv.org/abs/2303.13408) | parafraza psuje detektory | kod i model (11B) | skuteczność detekcji | |
| Attanasio i in. (2023). ferret. EACL demo. [ACL Anthology](https://aclanthology.org/2023.eacl-demo.29/) | IG, LIME, SHAP na modelu HF | `ferret-xai` | AOPC | |
| Morris i in. (2020). TextAttack. EMNLP demo. [ACL Anthology](https://aclanthology.org/2020.emnlp-demos.16/) | augmentery: WordNet, back-translation | `textattack` | - | |
| Jalili Sabet i in. (2020). SimAlign. Findings of EMNLP. [ACL Anthology](https://aclanthology.org/2020.findings-emnlp.147/) | wyrównanie słów bez treningu | `simalign` | alignment F1 | |
| Reimers, Gurevych (2019). Sentence-BERT. EMNLP. [ACL Anthology](https://aclanthology.org/D19-1410/) | filtr podobieństwa znaczenia | HF Hub | STS | |
| Sundararajan i in. (2017) IG [arXiv:1703.01365](https://arxiv.org/abs/1703.01365); Ribeiro i in. (2016) LIME [arXiv:1602.04938](https://arxiv.org/abs/1602.04938); Lundberg, Lee (2017) SHAP [arXiv:1705.07874](https://arxiv.org/abs/1705.07874) | badane metody XAI | Captum, `lime`, `shap` | - | |
