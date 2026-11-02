# Présentation Financial-MDL

Premier support en français : six diapositives principales pour sept minutes, puis quatre
annexes masquées dans le défilement normal pour les trois minutes de questions.

Le contenu et les notes sont dans [content.json](content.json). Les graphiques et le tableau
de sensibilité lisent les CSV de résultats versionnés. Les autres tableaux reprennent le
protocole et la spécification du codage. Tous sont des objets PowerPoint éditables.
Aucun modèle n'est entraîné et aucun score n'est recalculé pour créer le support.

Le message final distingue le gain de validation du compromis sur le test : 40% de variables
en moins, 112 bits économisés sur l'entraînement et une RMSE de test 0,77% plus élevée que
Ridge complet, dans la tolérance exploratoire de 2%.

Le fichier final versionné est `presentations/financial-mdl/financial-mdl-v1.pptx`.
Les rendus et contrôles de construction restent sous `artifacts/presentation-v1/build`.

## Reconstruction

L'outil de création utilise un environnement isolé, sans modifier les dépendances du benchmark :

```powershell
uv run --no-project --with python-pptx==1.0.2 python presentations/build_financial_mdl.py --output artifacts/presentation-replay/candidate.pptx
```

Sous Windows avec Microsoft PowerPoint installé, le script suivant exporte les dix rendus
sans ouvrir de fenêtre de présentation. Il teste aussi l'édition d'un graphique dans une copie,
puis son enregistrement et sa réouverture. Les éventuelles autres présentations ouvertes
restent intactes.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File presentations/render_powerpoint.ps1 -PresentationPath artifacts/presentation-replay/candidate.pptx -OutputDirectory artifacts/presentation-replay/render
```

Utiliser des chemins de sortie neufs pour conserver les versions précédentes.
