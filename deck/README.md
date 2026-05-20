# pitch-foresight

Deck de soutenance du POC ML Foresight. Storytelling honnête : produit → problème → solution ML → résultat assumé → insight produit.

**Ouvrir :** `index.html` dans un navigateur (ou `python -m http.server` puis `localhost:8000`). Fichier unique, zéro dépendance.

**Navigation :** `←` `→` ou `espace` · `F` plein écran · `Home`/`End`. Imprimable en PDF (chaque slide = une page) comme backup.

## Le fil narratif (12 slides)

1. Cover
2. Le produit — elevator pitch + carte signal Foresight
3. Le problème — la formule heuristique plafonne (~49 % winrate)
4. Les données — prod Hetzner → Postgres → export → 19 features (anti-leak allowlist)
5. La solution — classification binaire, 12 modèles, 3 repos
6. Le résultat sans filtre — aucun modèle ne bat l'heuristique
7. La leçon — pourquoi un seul split ment (variance 0.43→0.57)
8. Le piège n°2 — multiple-testing (economics +7 % = faux positif)
9. L'insight produit — l'edge est dans la latence, pas la prédiction 24 h
10. Ce que le projet apporte — rigueur ML + intelligence produit
11. La stack / démo live
12. Conclusion

## Le message

Le ML ne bat pas l'heuristique. C'est un résultat **rigoureux et assumé**, pas un échec : quand 12 modèles échouent identiquement, la limite est le signal (marché quasi-efficient à 24 h), pas le modèle. L'edge de Foresight est dans la vitesse de détection — le ML le confirme.

## Repos liés

- [ml-foresight](https://github.com/foresight-ml-poc/ml-foresight) — pipeline ML
- [backend-foresight](https://github.com/foresight-ml-poc/backend-foresight) — FastAPI
- [frontend-foresight](https://github.com/foresight-ml-poc/frontend-foresight) — démo React
- **pitch-foresight** (ici) — ce deck

Produit : [yourforesight.com](https://yourforesight.com)
