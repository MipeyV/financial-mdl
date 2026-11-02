# Trame de présentation — Financial-MDL

## Message central

Peut-on sélectionner une représentation plus courte en conservant une qualité prédictive
acceptable ? Nous avons implémenté un code MDL explicite pour des sous-ensembles de variables
et évalué ses choix sur des contrôles synthétiques puis sur une tâche de volatilité financière.
Les résultats actuels soutiennent une conclusion locale et exploratoire.

Format : **7 minutes de présentation, puis 3 minutes de questions**. Les sections suivantes
servent de notes ; un support de six diapositives peut suivre ce découpage :

| Diapositive | Contenu | Durée |
| --- | --- | ---: |
| 1 | Question de recherche | 30 s |
| 2 | Critère MDL et coûts encodés | 1 min |
| 3 | Protocole : entraînement, validation, test | 1 min |
| 4 | Contrôles synthétiques et lecture du graphique MDL–RMSE | 2 min |
| 5 | Résultats financiers : validation puis test final | 1 min 30 |
| 6 | Conclusion et limites | 1 min |

## 1. La question et le lien avec le cours

Partir du compromis entre coût du modèle et coût des données qu'il explique mal.
Écrire `L(sous-ensemble) + L(paramètres) + L(cibles | modèle, X)` et expliquer chaque terme.
Le critère de sélection utilise l'entraînement ; la RMSE hors entraînement évalue ses conséquences.

À dire : « Le nombre de variables est une mesure de taille. Notre objectif MDL tient aussi
compte du coût des paramètres et des erreurs restantes. »

## 2. Ce qui a été effectivement programmé

Un sélecteur exhaustif Ridge, incluant le modèle constant, avec un code d'identification
du sous-ensemble, des paramètres binary64 et des cibles quantifiées codées par un modèle
géométrique des résidus. Les hypothèses et la précision sont déclarées.

À dire : « Nous calculons une longueur de code explicite. Ce n'est ni la complexité exacte
de Kolmogorov, ni un compresseur complet de fichiers. X est une information partagée. »

## 3. Pourquoi commencer par le synthétique

La vérité est connue : deux signaux forts, une copie exacte et trois variables de bruit.
Deux autres scénarios contiennent du bruit seul ou un signal faible. Les cinq graines
évaluent la reproductibilité des observations. Chaque scénario utilise 600 lignes
d'entraînement et 300 lignes indépendantes d'évaluation dans l'expérience initiale.

L'objectif est de vérifier le comportement de la sélection, pas de présenter le synthétique
comme une simulation réaliste du marché.

## 4. Le graphique MDL–RMSE

Support : [vue complète](figures/mdl-frontier-v1/signal-all.svg), puis
[zoom](figures/mdl-frontier-v1/signal-zoom.svg). Les PNG correspondants sont disponibles dans
`artifacts/mdl-frontier-v1` pour l'insertion dans les diapositives.

- Chaque point est un sous-ensemble nommé, moyenné sur les mêmes cinq graines.
- Aller à gauche raccourcit le code d'entraînement ; descendre réduit la RMSE hors entraînement.
- La couleur indique le nombre de variables ; l'étoile repère le choix MDL sur entraînement.
- Le modèle sélectionné conserve les deux vrais signaux et enlève la copie et le bruit.
- La vue complète montre l'erreur des modèles privés d'un signal fort. Le zoom agrandit
  les faibles différences entre les modèles qui prédisent correctement.

Complément : `noise_only-zoom.png` et `weak_signal-zoom.png`. Le modèle constant est retenu
dans les deux cas à 600 observations. Pour le signal faible, cela ne signifie pas que la
variable est inutile : son gain de codage ne rembourse pas son coût dans cette expérience.

Les RMSE de tous les sous-ensembles sont des diagnostics exploratoires. Elles ne servent
pas à remplacer le choix MDL par le modèle qui obtient le meilleur score hors entraînement.
La tolérance +2% concerne le modèle sélectionné face à Ridge complet, pas toutes les combinaisons.

## 5. La sensibilité explique la limite

Support : tableau de [l'étude de sensibilité](mdl-sensitivity.md).

Le signal faible est écarté pour 150, 600 et 2 400 observations. À 9 600 observations,
il est retenu dans 5/5 graines avec les précisions 0.0001 et 0.01, et 2/5 avec la précision 0.1.
Il coûte 67 bits de représentation supplémentaires ; le gain de codage des cibles peut alors
rembourser ce coût. C'est un effet du volume de données et de la précision du code choisi.

À dire : « MDL peut écarter un vrai signal. Notre expérience montre dans quelles conditions
ce signal devient suffisamment utile pour justifier sa description. »

## 6. Le résultat financier — validation et test final

SPY quotidien : volatilité RMS des cinq prochaines séances, exprimée en unités annualisées.
Ce n'est pas une prévision à un an. Entraînement 2015–2021, validation 2022–2023 ; les labels
doivent se terminer dans leur partition. Tous les candidats utilisent les mêmes dates éligibles.

| Modèle | Variables utilisées | RMSE de validation | RMSE de test |
| --- | ---: | ---: | ---: |
| Persistance 20 séances | 1 | 0.069084 | 0.102860 |
| Ridge complet | 5 | 0.066996 | 0.078215 |
| Ridge + MDL | 3 | 0.065489 | 0.078818 |

Le modèle sélectionné utilise 40% de variables en moins et améliore la RMSE de validation
d'environ 2.25% face à Ridge complet. Le code total diminue de 112 bits, soit 0.56%.
Ces pourcentages décrivent des quantités différentes et doivent être annoncés séparément.

Les trois candidats figés ont ensuite été évalués sur les 497 origines communes du test
2024–2025, sans ajouter la validation à l'entraînement. MDL a une RMSE **0.77% plus élevée**
que Ridge complet sur le test, ce qui respecte la tolérance exploratoire **+2%**. L'amélioration
de validation ne se reproduit donc pas comme une amélioration sur test.

Support : [validation et test](figures/spy-final-test-v1/validation-test-rmse.svg) et
[erreurs finales](figures/spy-final-test-v1/test-metrics.svg). Les PNG sont disponibles dans
`artifacts/spy-final-test-v1/report`. Le [rapport final](spy-final-test.md) contient les vérifications.

À dire : « Le modèle sélectionné utilise 40% de variables en moins. Sur la période de test
réservée, sa RMSE augmente de 0.77%, dans notre tolérance déclarée de 2%. Le résultat soutient
un compromis de simplification, pas une amélioration prédictive générale. »

## 7. Conclusion et limites à savoir défendre

Le code choisi élimine bruit et redondance dans nos contrôles et fournit un modèle financier
plus petit, meilleur en validation et légèrement moins bon sur le test final. Une description plus courte
ne garantit pas une meilleure prédiction. Les paramètres binary64 sont coûteux ; la précision
et le modèle des résidus influencent les résultats. Les labels financiers se chevauchent et
les observations ne sont pas indépendantes. Cinq graines et une période de validation ne
permettent pas de revendiquer une supériorité générale.

Les KPI fondamentaux, la sélection par familles et les horizons calendaires mensuels/annuels
restent des extensions. Leur alignement aux dates réelles de publication est nécessaire.

## Organisation proposée pour les sept jours

1. Finaliser les graphiques et leur lecture.
2. Figer les candidats et exécuter une fois l'évaluation financière finale.
3. Consolider les tableaux et écrire la conclusion correspondant aux résultats obtenus.
4. Construire le support de présentation autour de la question et des expériences.
5. Préparer les réponses sur le code MDL, les baselines et les limites.
6. Répéter à durée réelle et simplifier les diapositives trop chargées.
7. Vérifier le support et les figures hors ligne, puis garder une copie exportée.

Les graphiques et l'évaluation finale sont réalisés. La priorité restante est de consolider
le support, puis de répéter la présentation en sept minutes et de préparer les trois minutes de questions.
