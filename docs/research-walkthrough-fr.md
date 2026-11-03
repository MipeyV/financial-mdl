# Comprendre Financial-MDL de A à Z

Ce guide explique l'expérience réalisée, ses choix et les conclusions que les résultats
autorisent. Les explications sont en français pour préparer une présentation en anglais.
Le [support](../presentations/financial-mdl/README.md) contient six diapositives principales
pour sept minutes, puis quatre annexes pour les questions. Les
[notes orales anglaises](../presentations/financial-mdl/speaker-notes-en.md) suivent ces slides.

## Parcours de lecture

Pour une première lecture, suivre les sections 1 à 10. Elles relient le problème au résultat.
Les sections 11 à 15 servent à retrouver les preuves, préparer l'oral et vérifier sa compréhension.
Les liens vers le code et les rapports permettent d'approfondir sans apprendre toute
l'architecture par cœur.

1. [La question scientifique](#1-la-question-scientifique)
2. [Ce que nous prédisons](#2-ce-que-nous-prédisons)
3. [Les variables et les modèles](#3-les-variables-et-les-modèles)
4. [La séparation temporelle](#4-la-séparation-temporelle)
5. [Le MDL et le lien avec le cours](#5-le-mdl-et-le-lien-avec-le-cours)
6. [Notre calcul des bits](#6-notre-calcul-des-bits)
7. [La sélection complète](#7-la-sélection-complète)
8. [Les expériences synthétiques](#8-les-expériences-synthétiques)
9. [Les résultats financiers](#9-les-résultats-financiers)
10. [Les conclusions et les limites](#10-les-conclusions-et-les-limites)
11. [Les fichiers et la reproductibilité](#11-les-fichiers-et-la-reproductibilité)
12. [MLflow et les contrôles logiciels](#12-mlflow-et-les-contrôles-logiciels)
13. [Le déroulé des sept minutes](#13-le-déroulé-des-sept-minutes)
14. [Les questions possibles en anglais](#14-les-questions-possibles-en-anglais)
15. [Vérifier sa compréhension](#15-vérifier-sa-compréhension)

## 1. La question scientifique

La question est : **un modèle choisi pour sa description plus courte conserve-t-il une
qualité prédictive acceptable sur de nouvelles observations ?**

Nous construisons une sélection de variables par Minimum Description Length (MDL).
Elle compare le coût de description des modèles au coût de description des cibles qui
restent à expliquer. Nous évaluons ensuite les prédictions avec des métriques habituelles.

Il faut donc distinguer trois quantités :

| Quantité | Question à laquelle elle répond | Usage ici |
| --- | --- | --- |
| Longueur MDL sur entraînement | Combien de bits exige notre convention de description ? | Choisir le sous-ensemble |
| Nombre de variables | Combien de colonnes utilise le modèle ? | Décrire sa simplification |
| RMSE hors entraînement | Quelle erreur fait-il sur d'autres observations ? | Évaluer le compromis prédictif |

Une réduction de variables peut augmenter l'erreur, la diminuer, ou presque la laisser
inchangée. L'expérience doit établir ce qui se produit. Elle n'a pas besoin de démontrer
que le MDL gagne partout pour apporter un résultat clair.

Notre règle exploratoire accepte une RMSE du modèle sélectionné au plus 2 % supérieure
à celle de Ridge complet. Le choix de 2 % exprime une tolérance pratique dans cette étude.
Nous ne l'avons pas dérivé d'un coût économique ou d'un test statistique.

Le travail comporte une implémentation du sélecteur, des contrôles synthétiques, une étude
de sensibilité et une évaluation financière finale. L'architecture du dépôt rend ces
expériences vérifiables. Elle soutient la question scientifique.

## 2. Ce que nous prédisons

### Une origine quotidienne et un horizon de cinq séances

Nous utilisons un snapshot figé de SPY, de 2015 à 2025. Une ligne correspond à une séance
observée. Après la clôture à la date t, le modèle dispose des informations jusqu'à cette
clôture et prédit la volatilité des **cinq séances suivantes**.

Si C_t est le cours de clôture et r_t = log(C_t / C_(t−1)), la cible est :

```text
y_t = sqrt((252 / 5) × (r_(t+1)² + ... + r_(t+5)²))
```

C'est une volatilité réalisée RMS : la racine de la moyenne des rendements au carré,
annualisée avec 252 séances. Nous ne retirons pas la moyenne des rendements. Ce choix
diffère d'un écart-type empirique avec centrage.

« Annualisée » indique l'unité de la volatilité. **L'horizon de prévision reste cinq séances.**
Une cible égale à 0.20 représente 20 % de volatilité annualisée, estimée sur cette fenêtre
future. Nous ne prédisons ni le rendement de demain ni le cours dans un an.

Si l'origine est un vendredi, on prend les cinq lignes de marché suivantes, en tenant
compte des lignes disponibles. Le code compte les séances observées, pas cinq jours
calendaires. La validation d'un calendrier boursier complet reste en amont du benchmark.

### Pourquoi les cibles se chevauchent

La cible d'aujourd'hui utilise les rendements des séances t+1 à t+5. Celle de demain utilise
t+2 à t+6. Quatre rendements sont communs. Les erreurs sur deux origines voisines peuvent
donc être dépendantes. Les 497 prévisions du test ne constituent pas 497 observations
indépendantes au sens d'une inférence statistique classique.

Source de la définition : [targets.py](../src/financial_mdl/targets.py) et
[protocole SPY](../protocols/spy-volatility-v1.json).

## 3. Les variables et les modèles

### Les cinq variables financières disponibles

Toutes les variables utilisent seulement les données disponibles après la clôture t.

| Variable | Définition | Famille |
| --- | --- | --- |
| `log_return_1` | log(C_t / C_(t−1)) | Marché |
| `log_return_5` | log(C_t / C_(t−5)) | Marché |
| `volume_change_1` | Volume_t / Volume_(t−1) − 1 | Marché |
| `realized_volatility_5` | RMS annualisée des cinq derniers rendements, dont r_t | Statistique |
| `realized_volatility_20` | RMS annualisée des vingt derniers rendements, dont r_t | Statistique |

Les volatilités explicatives regardent le passé. La volatilité cible regarde le futur.
Elles ne contiennent donc pas les mêmes rendements à une origine donnée.

Les familles fondamentales et complexité algorithmique existent comme catégories dans
l'architecture, sans variables implémentées dans cette expérience. Il n'y a ici ni KPI
comptable, ni sélection de groupes par famille. Le sélecteur choisit des colonnes individuelles.
Un futur KPI trimestriel devrait devenir disponible à sa date réelle de publication,
avec une règle pour ses révisions. La fréquence de publication d'un KPI et l'horizon de
la cible sont deux choix distincts.

### Un sous-ensemble est une combinaison de colonnes

Avec cinq variables, on peut garder aucune variable, une seule, deux, etc., jusqu'aux cinq.
Le nombre total de combinaisons vaut 2^5 = **32**. Le sous-ensemble vide correspond à un
modèle constant égal à la moyenne des cibles d'entraînement.

Par exemple, `{log_return_5, realized_volatility_20}` est un sous-ensemble de taille deux.
Un autre couple, `{log_return_1, volume_change_1}`, a la même taille mais apporte une
information différente. Des modèles avec le même nombre de variables peuvent donc avoir
des erreurs et des longueurs de description différentes.

Ne pas confondre les nombres : cinq variables financières donnent 32 modèles de
sous-ensembles. Les contrôles synthétiques ont six variables, donc 64 sous-ensembles.
Le test financier final compare trois approches figées.

### Pourquoi Ridge

Pour un ensemble de colonnes S, Ridge prédit :

```text
prediction_i = intercept + somme_j(coefficient_j × X_standardisé_ij)
```

Il ajuste les coefficients en minimisant une erreur quadratique pénalisée :

```text
somme_i(y_i − prediction_i)² + alpha × somme_j(coefficient_j²)
```

Ici alpha = 1 et l'intercept est activé. Le scaler soustrait la moyenne et divise par
l'écart-type calculés sur l'entraînement. Cette standardisation rend la pénalisation plus
comparable entre colonnes d'unités différentes. Les données de validation et de test
réutilisent ces statistiques d'entraînement.

La pénalisation Ridge réduit les coefficients. Elle ne les met généralement pas exactement
à zéro. La sélection explicite des colonnes vient de la couche MDL ajoutée autour de Ridge.
Un coefficient ou une importance exportée constitue un diagnostic du modèle, sans
interprétation causale automatique.

La cible est positive, mais Ridge ne contraint pas ses sorties à être positives. Le benchmark
n'applique pas de clipping implicite aux prévisions négatives. Ajouter cette opération
constituerait un changement de candidat à définir et à évaluer explicitement.

Ridge est une référence simple et adaptée à une expérience où il faut décrire les paramètres.
Nous n'avons pas démontré qu'il est la meilleure architecture possible. Comparer des arbres,
Lasso, modèles de volatilité ou réseaux demanderait des réglages et des codes de modèle
comparables. L'adaptateur Random Forest existe dans le framework, sans résultat établissant
sa supériorité dans cette étude. Le code MDL actuel décrit spécifiquement un Ridge scalaire.

### Les références de prédiction

La moyenne d'entraînement prédit une constante. Elle donne une référence minimale.
La persistance à cinq ou vingt séances prédit la volatilité passée correspondante.
Elle ne réapprend pas de relation à partir des cibles. La persistance 20 utilise une fenêtre
passée de vingt séances pour prédire la cible future à cinq séances : le 20 ne change pas
l'horizon futur.

La moyenne n'est pas une prédiction aléatoire. Un modèle proche de son erreur n'apporte
simplement pas de gain mesuré contre cette référence sur les observations comparées.
La persistance est plus informative pour une série dont la volatilité présente une inertie.

Implémentation : [features.py](../src/financial_mdl/features.py),
[models.py](../src/financial_mdl/models.py), [candidates.py](../src/financial_mdl/candidates.py).

## 4. La séparation temporelle

| Partition | Période | Origines éligibles | Rôle |
| --- | --- | ---: | --- |
| Entraînement | 2015–2021 | 1 738 | Scaler, coefficients, distribution des résidus, choix MDL |
| Validation | 2022–2023 | 496 | Comparaison et analyse exploratoire |
| Test final | 2024–2025 | 497 | Évaluation des trois candidats figés |

Tous les candidats d'une phase partagent la même cible, les mêmes dates et les mêmes
partitions. Les données doivent être complètes sur les **cinq variables du protocole avant
sélection**, même si un candidat n'en consomme ensuite qu'une ou trois. Cela évite qu'un
modèle paraisse meilleur simplement parce qu'il évite certaines dates difficiles.

Les fenêtres de cible se terminent à l'intérieur de leur partition. Une origine de fin
2021 dont les cinq rendements futurs dépassent la limite de l'entraînement est écartée.
Cette purge empêche les cibles d'entraînement d'utiliser des rendements de validation.
Le protocole utilisé a un gap supplémentaire de zéro séance. Il conserve les dépendances
temporelles internes à chaque partition.

Le test final utilise les mêmes ajustements sur 2015–2021 que la validation. **Nous
n'ajoutons pas la validation à l'entraînement.** Cela permet d'évaluer les candidats déjà
étudiés. À une origine de test, on peut calculer une variable avec les cours de test déjà
observés à cette date. Les cibles de test ne servent jamais à réajuster les paramètres.

### Pourquoi garder un test si la validation est commune

Une validation commune rend les modèles comparables. Au fil des analyses, les décisions
du chercheur peuvent cependant s'adapter à cette validation : choix de modèles, de
codage, d'hyperparamètres ou du récit. Une dernière période réservée apporte une évaluation
des choix figés après cette exploration.

Avant de demander les scores de test, nous avons versionné un
[plan](../experiments/spy-final-test-v1.json) fixant Ridge complet, Ridge + MDL et
persistance 20, ainsi que les règles de comparaison. Le commit du plan est `ebae61c`.
Les trois évaluations finales ont ensuite utilisé ce plan, sans réglage depuis les scores.

Le test a désormais été consulté. Rejouer exactement le calcul vérifie la reproductibilité.
Ce rejeu ne produit pas un nouveau test indépendant. Une nouvelle recherche guidée par
ces résultats demande un nouveau plan d'évaluation.

## 5. Le MDL et le lien avec le cours

L'idée du MDL est de préférer une explication qui décrit efficacement le modèle **et**
les données restant à expliquer. Un modèle très complexe peut réduire les erreurs tout
en coûtant cher à transmettre. Un modèle constant coûte peu à décrire, mais peut laisser
beaucoup d'information dans les résidus.

On peut imaginer un émetteur qui veut permettre à un destinataire de reconstruire les
cibles. Les deux connaissent déjà les variables X et les règles de codage. L'émetteur
doit préciser les colonnes retenues, les paramètres du modèle et les corrections nécessaires
aux prédictions. Nous cherchons la description totale la plus courte selon cette convention.

Notre critère est :

```text
L_total = L(S) + L(paramètres | S, X) + L(cibles | paramètres, S, X)
```

Les deux premiers termes décrivent le modèle. Le dernier décrit les cibles avec ce modèle.
Cela constitue un code en deux parties, même si nous détaillons le premier en deux termes.

Dans le chapitre 4 du cours, les diapositives 20–21 relient coût du modèle et coût des
données. Les diapositives 24–28 discutent notamment les probabilités, les longueurs
−log₂ P et la précision finie. Notre usage d'un Ridge, de mots binary64 et d'un modèle
géométrique des résidus est une instanciation explicite de ces idées. Les diapositives du
cours ne prescrivent pas ces choix particuliers.

La complexité de Kolmogorov concerne la longueur du plus court programme produisant un
objet, et elle n'est pas calculable en général. Nous calculons une longueur dans un code
défini et limité. Nous ne trouvons pas le plus court programme de tous les programmes possibles.

Le MDL dépasse la sélection de variables : il peut comparer des familles de modèles ou
d'autres représentations. Dans ce projet, la famille Ridge est fixée et la représentation
varie avec les colonnes conservées.

## 6. Notre calcul des bits

### 6.1 Ce que le destinataire connaît déjà

Le code est conditionnel à la matrice d'entraînement X, aux noms et à l'ordre des colonnes,
au nombre de lignes et à la spécification commune. Nous ne comptons pas le stockage des
cours, de X ou des programmes de construction des variables.

Les moyennes et écarts-types du scaler se déduisent de ce X connu. Nous n'ajoutons donc
pas leurs mots binary64. Ce raisonnement concerne une description conditionnelle des
cibles. Pour mesurer la taille d'un paquet de déploiement autonome, il faudrait inclure
les informations absentes du destinataire, dont les statistiques du scaler.

### 6.2 Identifier les variables : L(S)

Avec p colonnes possibles et k retenues :

```text
L(S) = ceil(log₂(p + 1)) + ceil(log₂(comb(p, k)))
```

`ceil` signifie arrondir à l'entier supérieur. Le premier champ indique combien de
variables sont retenues. Pour p=5, k peut valoir 0 à 5 : six possibilités demandent trois
bits. Le second champ indique quelle combinaison de taille k est choisie, par son rang
dans un ordre fixé.

Pour p=5 et k=3, il existe dix triplets. Quatre bits permettent de distinguer ces dix cas,
avec quelques motifs inutilisés. Le total vaut **3 + 4 = 7 bits**.
Pour k=5, il n'existe qu'un ensemble contenant toutes les colonnes. Le rang demande
zéro bit, donc **3 + 0 = 3 bits**.

Identifier un sous-ensemble peut ainsi coûter davantage qu'annoncer « toutes les colonnes ».
Le gain vient ensuite du nombre de coefficients à décrire. Nous ne comptons pas les
valeurs de toutes les lignes pour identifier les noms des colonnes.

### 6.3 Décrire les paramètres

Nous transmettons k coefficients, un intercept et un paramètre q de la distribution
des résidus. Chacun utilise 64 bits au format IEEE-754 binary64 :

```text
L(paramètres | S, X) = 64 × (k + 2)
```

Cela donne 448 bits pour k=5 et 320 bits pour k=3. Même le modèle sans variable a
l'intercept et q, donc 128 bits. Le code sérialise réellement ces nombres avec `struct.pack`
et conserve leurs octets dans `parameter_hex`.

Le choix de 64 bits est simple et explicite. Il est coûteux et influence la sélection.
Nous ne prétendons pas que chaque coefficient nécessite scientifiquement 64 bits de
précision utile. Un codage plus économique pourrait conserver davantage de variables.

### 6.4 Transformer les cibles en symboles entiers

Décrire exactement un réel arbitraire nécessiterait une précision non bornée. Nous fixons
une grille de précision delta = 0.0001, avant l'évaluation. Dans nos unités de volatilité,
cela correspond à 0.01 point de pourcentage.

```text
z_i = rint(y_i / delta)
m_i = rint(prediction_i / delta)
r_i = z_i − m_i
```

`rint` arrondit au plus proche, avec les égalités vers l'entier pair. Le résidu entier est
la différence des deux symboles arrondis. Ce n'est pas toujours identique à arrondir
directement `(y_i − prediction_i) / delta`.

Exemple illustratif, distinct des données SPY : pour y=0.1200 et une prédiction 0.1198,
on obtient z=1200, m=1198 et r=2. Le destinataire calcule m, reçoit la correction r,
retrouve z=m+r et représente la cible par delta × z. La reconstruction conserve les
symboles quantifiés, avec une erreur d'arrondi d'au plus delta/2 sur la valeur initiale.

Le modèle s'ajuste aux cibles continues originales. La quantification intervient dans
le code MDL. Les métriques finales utilisent également les cibles continues originales.

### 6.5 Donner une probabilité aux corrections

Les résidus entiers suivent, pour le calcul du code, une distribution géométrique bilatérale :

```text
P(r | q) = ((1 − q) / (1 + q)) × q^abs(r), pour 0 ≤ q < 1
```

Les petits résidus reçoivent davantage de probabilité. Une grande correction coûte plus
de bits. Une probabilité 1/8 correspond à −log₂(1/8)=3 bits idéaux.
À titre d'exemple, q=1/2 donne P(r=2)=1/12, soit environ 3.585 bits idéaux.

Pour chaque modèle, nous estimons q sur ses seuls résidus d'entraînement :

```text
a = moyenne(abs(r_i))
q = a / (sqrt(a² + 1) + 1)
```

Le paramètre q est inclus dans les mots binary64. Si tous les résidus sont nuls, q=0
attribue une probabilité un à la séquence de zéros. Sinon, le coût des cibles est :

```text
L(cibles | modèle) = ceil(−somme_i log₂ P(r_i | q))
```

Nous arrondissons le coût de la séquence entière, une seule fois. Les longueurs proviennent
des probabilités sous le modèle choisi. Nous calculons ces longueurs sans produire le
flux binaire complet des résidus. L'expérience n'est donc pas une mesure de la taille
d'un fichier ZIP. L'hypothèse factorisée des probabilités n'établit pas l'indépendance
réelle des erreurs temporelles.

La spécification de référence est [mdl-coding.md](mdl-coding.md) et le code est
[mdl.py](../src/financial_mdl/mdl.py).

## 7. La sélection complète

Le pseudo-code suivant décrit les étapes réellement effectuées :

```text
Entrées : X_train, y_train, alpha=1, delta=0.0001, convention de codage fixée
Calculer les statistiques de standardisation à partir de X_train
Pour chacun des 2^p sous-ensembles S, y compris l'ensemble vide :
    Ajuster un Ridge sur les colonnes S de l'entraînement
    Pour l'ensemble vide, prédire la moyenne d'entraînement
    Calculer les résidus quantifiés et estimer q sur l'entraînement
    Compter les bits du sous-ensemble, des paramètres et des cibles
Retenir le sous-ensemble de coût total minimal
En cas d'égalité : préférer moins de variables, puis l'ordre initial des colonnes
Ajuster le même Ridge et le même scaler sur les colonnes retenues de l'entraînement
Prédire les observations de la phase demandée
Calculer MAE et RMSE sur ces observations
Comparer RMSE_MDL / RMSE_full à 1.02
```

La recherche MDL ne consulte aucun score de validation ou de test. Ridge optimise son
objectif quadratique régularisé à l'intérieur de chaque sous-ensemble. Le sélecteur MDL
compare ensuite les coûts de ces Ridge ajustés. Il n'optimise pas conjointement tous les
paramètres réels pour minimiser directement la longueur du code. En particulier, notre
code de résidus géométriques dépend des erreurs absolues quantifiées, ce qui diffère de
l'objectif quadratique interne de Ridge.

La tolérance de RMSE intervient après la sélection. Elle n'ajoute pas un terme au MDL
et ne permet pas de changer de sous-ensemble après lecture du test. Un dépassement de
2 % aurait constitué un échec de cette tolérance, à rapporter tel quel.

Une recherche exhaustive est accessible pour cinq ou six variables. Son coût augmente
exponentiellement. Le sélecteur limite par défaut l'univers à douze variables. Pour de
très grands univers, il faudrait étudier une stratégie de recherche différente et ses limites.

## 8. Les expériences synthétiques

### 8.1 Pourquoi des données artificielles

Sur SPY, nous ignorons quelles variables sont réellement utiles dans tous les régimes.
Un générateur contrôlé permet de connaître la relation qui produit y. On peut donc vérifier
si le sélecteur enlève du bruit, évite une redondance ou rate un signal faible.

Les six colonnes sont `signal_a`, `signal_b`, une copie exacte de `signal_a` et trois
variables de bruit. Les colonnes de base sont normales standard et le bruit de cible a
un écart-type de 0.4. Nous utilisons cinq graines, 600 lignes d'entraînement et 300 lignes
séparées pour l'évaluation de départ.

| Scénario | Cible générée | Choix MDL observé dans les cinq graines |
| --- | --- | --- |
| Signaux forts | 1 + 2a − 1.5b + bruit | Deux variables informatives, sans doublon |
| Bruit seul | 1 + bruit | Modèle constant |
| Signal faible | 1 + 0.05a + bruit | Modèle constant à n=600 |

La copie exacte de a contient la même information. Retrouver le groupe informatif sans
redondance est le critère pertinent, plutôt qu'un nom de colonne unique à tout prix.

Le signal faible est véritablement prédictif dans le générateur. Son exclusion montre
une limite du coût choisi : sur cet échantillon, son bénéfice ne rembourse pas la description
d'un coefficient et de sa présence dans le sous-ensemble.

### 8.2 Lire les graphiques MDL–RMSE

L'[étude de tous les sous-ensembles](mdl-frontier.md) évalue les 64 combinaisons de chaque
scénario, sur cinq graines : 64 × 3 × 5 = **960 évaluations de sous-ensembles**.

Dans la présentation, l'axe horizontal donne le coût MDL moyen **d'entraînement**, l'axe
vertical la RMSE moyenne **hors entraînement**, et la couleur le nombre k de variables.
Chaque point représente une combinaison nommée, moyennée sur les cinq graines. Plusieurs
points d'une même couleur correspondent à des combinaisons différentes de même taille.
Certaines se superposent à cause de la copie exacte.

Le carré magenta du PowerPoint montre le choix MDL, effectué séparément dans chaque
graine. Le carré vide montre le modèle complet. Les figures SVG/HTML originales utilisent
une étoile pour le choix MDL. Le symbole change selon le support, pas la règle de sélection.

La vue complète révèle les fortes erreurs des modèles privés d'un signal. Le zoom de la
slide principale agrandit les écarts minuscules entre les modèles qui prédisent correctement.
Le zoom ne doit pas faire passer quelques dix-millièmes de RMSE pour un gain majeur.

Un sous-ensemble incluant du bruit peut avoir une RMSE hors entraînement légèrement plus
basse par hasard. Par exemple, dans les signaux forts, le choix MDL a une RMSE moyenne
de 0.406789, et un triplet avec une variable de bruit atteint environ 0.406669. Nous
n'utilisons pas cette observation pour remplacer le choix MDL.

Les scores synthétiques et financiers concernent des cibles d'unités différentes.
Comparer directement 0.406789 à 0.078818 n'a pas de sens.

### 8.3 Quand le signal faible devient rentable

L'[étude de sensibilité](mdl-sensitivity.md) fixe une grille avant calcul : tailles
d'entraînement 150, 600, 2 400 et 9 600 ; précisions 0.0001, 0.01 et 0.1 ; cinq graines
et trois scénarios. Cela donne **180 recherches MDL** et leurs références complètes.
Les blocs d'entraînement sont emboîtés, avec un bloc d'évaluation de 300 lignes fixe.

| Taille d'entraînement | Signal faible retenu, delta=0.0001 | delta=0.01 | delta=0.1 |
| ---: | ---: | ---: | ---: |
| 150 | 0/5 | 0/5 | 0/5 |
| 600 | 0/5 | 0/5 | 0/5 |
| 2 400 | 0/5 | 0/5 | 0/5 |
| 9 600 | 5/5 | 5/5 | 2/5 |

Avec six colonnes, passer du modèle vide à une variable coûte 64 bits de coefficient
et trois bits supplémentaires de sous-ensemble, soit **67 bits**. À n=9 600 et
delta=0.0001, le signal économise de 99 à 137 bits sur les cibles, donc rembourse ce coût.
Avec delta=0.1, seules deux graines le remboursent.

La taille d'échantillon et la précision du code influencent ainsi le choix. Le faible
signal n'est pas absent aux petites tailles. Il est trop coûteux pour cette convention
sur ces données. Dans cette grille, les choix MDL respectent tous la tolérance de 2 %
face à leurs Ridge complets, sans gagner systématiquement en RMSE.

Les générateurs de l'étude initiale et de la sensibilité ne consomment pas l'aléatoire
dans le même ordre. On interprète chaque étude dans son propre protocole, sans attribuer
leurs écarts absolus de RMSE au seul changement de n.

## 9. Les résultats financiers

### 9.1 Le sous-ensemble retenu et son coût

Le sélecteur conserve :

- `log_return_5` ;
- `realized_volatility_5` ;
- `realized_volatility_20`.

Il enlève `log_return_1` et `volume_change_1` dans cette expérience. Ce résultat ne démontre
pas leur inutilité universelle ou causale.

| Modèle | Bits d'identification | Bits de paramètres | Bits des cibles | Total sur entraînement |
| --- | ---: | ---: | ---: | ---: |
| Ridge complet, 5 variables | 3 | 448 | 19 706 | 20 157 |
| Ridge + MDL, 3 variables | 7 | 320 | 19 718 | 20 045 |

Retirer deux coefficients économise 128 bits. L'identification du sous-ensemble coûte
quatre bits supplémentaires et les cibles douze bits de plus. Le gain net vaut :

```text
128 − 4 − 12 = 112 bits
```

Le modèle sélectionné laisse donc les cibles un peu plus difficiles à coder, mais réduit
plus fortement sa représentation. C'est précisément le compromis recherché.

Les pourcentages répondent à des dénominateurs différents :

| Mesure de réduction | Calcul | Résultat |
| --- | --- | ---: |
| Nombre de variables | (5 − 3) / 5 | 40 % |
| Représentation du modèle, sous-ensemble inclus | (451 − 327) / 451 | 27.49 % |
| Code total d'entraînement | 112 / 20 157 | 0.56 % |

Dire « 40 % de compression totale » serait donc incorrect. Les cibles dominent la
longueur totale. Ces bits décrivent l'entraînement et ne mesurent pas une compression du test.

### 9.2 Les métriques de prédiction

```text
MAE  = moyenne(abs(y_i − prediction_i))
RMSE = sqrt(moyenne((y_i − prediction_i)²))
```

Les deux métriques sont meilleures lorsqu'elles diminuent. La RMSE donne plus de poids
aux grandes erreurs. La MAE complète son interprétation. Elles gardent l'unité de la cible.
Une RMSE de 0.078215 correspond à environ 7.82 points de pourcentage de volatilité
annualisée. Ce n'est pas une erreur relative de 7.82 % sur chaque observation.

La MAPE divise chaque erreur absolue par la cible. Elle devient instable quand la cible
approche zéro et change le poids des périodes de faible volatilité. Elle pourrait compléter
une autre étude avec une justification explicite. Le protocole actuel utilise RMSE et MAE.

### 9.3 Validation et test, séparément

| Modèle | RMSE de validation 2022–2023 | RMSE de test 2024–2025 | MAE de test |
| --- | ---: | ---: | ---: |
| Ridge complet | 0.066996 | 0.078215 | 0.046150 |
| Ridge + MDL | 0.065489 | 0.078818 | 0.046349 |
| Persistance 20 séances | 0.069084 | 0.102860 | 0.058475 |

Les chiffres arrondis viennent du [CSV versionné](results/spy-final-test-v1-summary.csv).
Le [rapport final](spy-final-test.md) conserve davantage de décimales et les vérifications.
Les premiers résultats incluent aussi moyenne et persistance 5 sur validation dans
[mdl-results.md](mdl-results.md). Ces deux références ne font pas partie des trois
candidats du test final figé.

Sur validation, la variation relative de RMSE du modèle MDL face au complet vaut :

```text
100 × (RMSE_MDL / RMSE_full − 1) ≈ −2.25 %
```

Sur test, le même calcul donne **+0.77 %**. Le ratio vaut environ 1.00770853, inférieur
à 1.02. La différence absolue est environ 0.000603 en unités de cible, soit 0.0603 point
de pourcentage de volatilité annualisée. Le +0.77 % exprime l'augmentation **relative
de la RMSE**, ce qui constitue une autre quantité.

Sur cette période, Ridge complet a les meilleures RMSE et MAE des trois candidats.
Le modèle MDL utilise moins de variables et respecte la tolérance. Tous deux battent
la persistance 20 ; la RMSE du MDL est environ 23.37 % inférieure à celle de cette référence.
L'amélioration observée sur validation ne se reproduit pas comme amélioration sur test.

Chaque colonne est une comparaison sur des observations alignées. Le changement de niveau
entre validation et test porte aussi sur des périodes de marché différentes. Il n'isole
pas un effet causal du modèle ou une modification de l'algorithme.

## 10. Les conclusions et les limites

La conclusion défendable est : **ce code sélectionne un Ridge plus petit dont l'erreur
sur la période de test reste dans la tolérance exploratoire déclarée, avec un léger coût
prédictif face au Ridge complet**.

Les contrôles donnent aussi un enseignement : le code enlève le bruit et la redondance
dans les scénarios étudiés, mais peut supprimer un vrai signal faible. La sensibilité
montre comment davantage de données peut rendre ce signal rentable en bits.

Les limites qui affectent cette conclusion sont les suivantes :

- Le codage binary64, la grille et la distribution des résidus sont des choix expérimentaux.
  Une autre convention peut sélectionner autre chose.
- La recherche compare des Ridge ajustés avec alpha fixé. Elle n'établit pas un optimum
  universel parmi toutes les architectures, tous les paramètres et tous les codes.
- Le coût est conditionnel à X connu. Il ne mesure pas le coût complet d'acquisition,
  de calcul des variables ou de déploiement.
- Une série financière et une seule période de test donnent une portée empirique limitée.
  Les cibles se chevauchent et nous ne fournissons pas de preuve de supériorité statistique.
- Le seuil de 2 % est exploratoire. Le respecter n'établit pas une non-infériorité statistique.
- Le snapshot fournisseur est figé pour reproduire les résultats, sans certification
  historique point-in-time de toutes ses valeurs et corrections.

Une suite raisonnable serait de fixer un autre code de paramètres et d'évaluer sa
sensibilité selon un nouveau protocole. Étendre à d'autres séries ou à des blocs temporels
prédéfinis aiderait à étudier la robustesse. Ces extensions ne sont pas déjà réalisées.

## 11. Les fichiers et la reproductibilité

### Le trajet d'une expérience dans le code

| Étape | Fichier principal | Ce qu'il faut comprendre |
| --- | --- | --- |
| Entrée CLI et candidats | [cli.py](../src/financial_mdl/cli.py), [candidates.py](../src/financial_mdl/candidates.py) | Les choix du modèle restent séparés des règles du benchmark |
| Données et protocole | [data.py](../src/financial_mdl/data.py), [protocol.py](../src/financial_mdl/protocol.py) | Le protocole se lie à un snapshot précis |
| Variables et cible | [features.py](../src/financial_mdl/features.py), [targets.py](../src/financial_mdl/targets.py) | Passé pour X, futur pour y |
| Frontières temporelles | [splitting.py](../src/financial_mdl/splitting.py) | Les labels se terminent dans leur partition |
| Calcul et choix MDL | [mdl.py](../src/financial_mdl/mdl.py) | Tous les coûts de sélection viennent de l'entraînement |
| Ajustement et prédictions | [benchmark.py](../src/financial_mdl/benchmark.py), [forecasting.py](../src/financial_mdl/forecasting.py) | Les contrats conservent dates, horizons et dimensions explicites |
| Mesures et exports | [metrics.py](../src/financial_mdl/metrics.py), [reporting.py](../src/financial_mdl/reporting.py) | Les observations comparées doivent correspondre |
| Suivi optionnel | [tracking.py](../src/financial_mdl/tracking.py) | MLflow enregistre les résultats sans définir les modèles |

Un protocole fixe notamment le snapshot, les variables disponibles, la cible, les dates,
les frontières et les métriques. Un candidat fixe notamment l'estimateur, le sélecteur et
ses paramètres. Changer de sous-ensemble n'autorise pas à déplacer les dates ou à redéfinir
la cible pour ce candidat.

Les empreintes SHA-256 identifient données, protocole et observations évaluées.
Elles aident à détecter des différences d'entrées. Elles ne prouvent pas à elles seules
que les données sont exemptes de biais ou que la méthode est scientifiquement optimale.

### Où retrouver les preuves

| Besoin | Fichier ou dossier |
| --- | --- |
| Convention de codage | [mdl-coding.md](mdl-coding.md) |
| Premiers contrôles et validation | [mdl-results.md](mdl-results.md) |
| Sensibilité au volume et à la précision | [mdl-sensitivity.md](mdl-sensitivity.md) |
| Lecture des 64 combinaisons | [mdl-frontier.md](mdl-frontier.md) |
| Plan du test avant scores | [spy-final-test-v1.json](../experiments/spy-final-test-v1.json) |
| Résultats et audit du test | [spy-final-test.md](spy-final-test.md), [verification.json](results/spy-final-test-v1-verification.json) |
| Figures versionnées | [frontières](figures/mdl-frontier-v1), [test final](figures/spy-final-test-v1) |
| Sorties détaillées locales | `artifacts/spy-mdl-v1`, `artifacts/mdl-frontier-v1`, `artifacts/mdl-sensitivity-v1`, `artifacts/spy-final-test-v1` |
| Support et texte oral | [dossier de présentation](../presentations/financial-mdl/README.md) |

Les dossiers `data` et `artifacts` sont ignorés par Git. Une personne qui clone le dépôt
n'obtient donc pas automatiquement le snapshot ni la base MLflow locale. Les tableaux
et figures versionnés restent consultables. Une nouvelle récupération fournisseur peut
produire une autre empreinte et nécessite un protocole adapté.

Chaque run exporte notamment `run.json`, `protocol.json`, `metrics.json` et
`predictions.parquet`. Les runs MDL ajoutent `selection.json` et `mdl_candidates.parquet`,
où l'on retrouve les 32 coûts financiers. Les contrôles du test vérifient que tous ces
coûts et paramètres d'entraînement restent identiques à ceux de la validation.

Pour consulter les résultats, commencer par les rapports et les CSV. Recalculer les
figures depuis les sorties existantes ne réentraîne pas de modèle. Les commandes de
reproduction scientifique sont dans les rapports et exigent des sorties neuves.

## 12. MLflow et les contrôles logiciels

MLflow fournit une interface de suivi des expériences : paramètres, métriques, informations
de provenance et liens vers les artefacts. Notre installation conserve les métadonnées
dans SQLite et les fichiers sur disque local sous `artifacts/mlflow`.

Depuis la racine du dépôt, avec l'extra tracking installé :

```powershell
uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db
```

La commande affiche l'adresse locale à ouvrir dans le navigateur. Les coûts se trouvent
sous `training/mdl/...`, les erreurs sous `validation/...` ou `test/...`. Vérifier la phase
avant toute comparaison. Les expériences synthétiques autonomes conservent leurs sorties
dans `artifacts`, sans être toutes enregistrées dans MLflow. Les trois candidats financiers
finaux et leur comparaison ont chacun un run local terminé.

MLflow apporte de la traçabilité. La validité temporelle vient du protocole et de son
implémentation. Le suivi actuel démarre en fin d'exécution, donc certaines erreurs survenant
plus tôt ne produisent pas de run d'échec dans MLflow. Le setup local répond aux besoins
de l'étude, sans constituer un service d'équipe hébergé avec sauvegardes et contrôle d'accès.

Le fichier `uv.lock` fixe les dépendances. GitHub Actions lance Ruff, les builds, des tests
hors ligne sur plusieurs versions de Python et Windows/Linux, ainsi qu'une intégration
MLflow locale. Les tests couvrent notamment les frontières temporelles, les formes des
prédictions et le code MDL. Ils vérifient des règles logicielles ; ils ne garantissent pas
qu'une hypothèse financière est vraie. Le workflow est décrit dans
[CONTRIBUTING.md](../CONTRIBUTING.md).

## 13. Le déroulé des sept minutes

| Slide | Temps visé | Message à expliquer |
| --- | ---: | --- |
| MDL feature selection | 0:00–0:30 | La question du modèle plus court à performance acceptable |
| The MDL selection criterion | 0:30–1:30 | Le total de bits sélectionne, avec l'exemple des 112 bits |
| A common temporal protocol | 1:30–2:30 | Les mêmes dates, l'entraînement seul, puis validation et test |
| Synthetic controls | 2:30–4:30 | Les signaux connus, la lecture du nuage et la limite du signal faible |
| Final test results | 4:30–6:00 | 40 % de variables en moins, RMSE de test +0.77 % |
| Conclusion and limitations | 6:00–7:00 | Un compromis observé, dépendant du code et du protocole |

Les quatre annexes sont masquées au défilement normal. Elles servent à répondre sur
l'identification des variables, le signal faible, la vue complète du nuage et la portée
de la conclusion. On peut ouvrir une slide masquée via le mode présentateur ou son numéro.

Les notes anglaises sont un point de départ à reformuler. Répéter avec un chronomètre :
un budget de 420 secondes ne garantit pas une durée orale exacte. Si le temps manque,
raccourcir la description du nuage avant de retirer les frontières temporelles ou la
distinction validation/test.

La phrase centrale peut être :

> With this explicit code, MDL selects three of five features. Test RMSE increases by
> 0.77 percent relative to full Ridge, within our predeclared two percent tolerance.

Pour les nombres en anglais : `0.77%` se lit « zero point seven seven percent » et
`0.078818` « zero point zero seven eight eight one eight ». À l'oral, le pourcentage relatif
et la comparaison suffisent généralement, sans lire toutes les décimales.

## 14. Les questions possibles en anglais

**What does MDL optimise here?**

It minimises training description length: the feature subset, binary64 parameters and
quantised-target code. We evaluate the selected model separately with held-out RMSE and MAE.

**Are you simply selecting fewer features?**

Feature count affects the parameter cost. Target coding also depends on how well those
features explain the data. Two subsets with the same size can have different total lengths.

**Do you minimise MDL and RMSE together?**

The selector minimises training MDL. The two percent RMSE tolerance is a separate evaluation
rule applied afterwards. It does not enter the subset search.

**Why does the smaller model need more subset-identification bits?**

With all five features, the subset is unique. With three of five, we must identify one of
ten possible triples. The smaller model saves more bits on coefficients than it adds elsewhere.

**Why do you need to send q?**

The decoder uses q to assign probabilities to residuals. We estimate it from the training
residuals and include its 64-bit representation in the model cost.

**Do you count the scaler parameters?**

We condition on known training X. Its means and scales can be derived by the decoder.
A standalone deployment code with unknown training X would need a different accounting.

**Have you implemented a compressor?**

We serialise the parameter words and calculate residual code lengths from a specified
probability distribution. We do not produce the complete residual bitstream.

**Is this exact Kolmogorov complexity?**

We use a computable reference code within a fixed model family. Exact Kolmogorov complexity
is uncomputable in general, and our experiment makes no claim to compute it.

**Why 64 bits per parameter? Would 32 bits change the result?**

Binary64 gives an explicit parameter representation, but it is expensive. Changing the
precision could change selection and predictions. We would specify and test a new code
under a new evaluation plan.

**Why can MDL discard a real signal?**

Its reduction in target code length may be too small to repay its representation cost.
Our weak-signal study shows this at small sample sizes and shows retention at larger sizes.

**Why not choose the point with the lowest held-out RMSE?**

That would change the selection rule after inspecting the diagnostic scores. We keep the
training MDL choice and report its predictive performance.

**Is MDL better than full Ridge?**

On validation its RMSE is lower. On the final test its RMSE is 0.77 percent higher, while
it uses forty percent fewer features. Our result supports this simplification trade-off.

**Does passing two percent prove non-inferiority?**

It meets our exploratory tolerance on this period. We did not perform a statistical
non-inferiority test, and overlapping targets create dependence between observations.

**How did you avoid leakage?**

Features use information available at the origin. Training target windows end inside
training. We fit selection, scaling and coefficients on training only. All candidates
share eligible evaluation rows, and we froze the final plan before test scoring.

**Why have validation and test if MDL only uses training?**

Validation supports the wider experimental development and analysis. A final reserved
period evaluates the frozen choices after that exploration.

**Why Ridge rather than a neural network or a volatility model?**

Ridge provides a simple, inspectable reference for this coding experiment. We have not
established it as the best architecture. Comparing other families requires explicit model
codes, fair tuning rules and a new evaluation design.

**Where are the quarterly fundamentals or monthly forecasts?**

They remain extensions. This study uses daily market features and a five-session target.
Fundamentals would require publication-time alignment, including revision handling.

**What is your next experiment?**

One useful next experiment would compare alternative parameter codes using a predeclared
protocol, then evaluate robustness on additional series or reserved temporal blocks.

**Did you use an AI assistant?**

Si la question est posée, décrire fidèlement l'aide réellement utilisée et les règles de
l'école. Préparer des exemples précis de ce que tu as vérifié et compris. Le support, le
code ou le récit ne doivent pas inventer une autonomie de réalisation ou des validations
qui n'ont pas eu lieu. L'essentiel à défendre ici reste la démarche scientifique effective.

## 15. Vérifier sa compréhension

Avant de répéter l'oral, essayer de répondre sans relire :

1. Quelle information est disponible à l'origine t et quels rendements forment y_t ?
2. Pourquoi une volatilité annualisée à cinq séances n'est-elle pas une prévision annuelle ?
3. Pourquoi tous les modèles gardent-ils le même filtre sur les cinq variables ?
4. Pourquoi p=5 et k=3 donnent-ils sept bits d'identification ?
5. Quels sont les k+2 nombres transmis sur 64 bits ?
6. Comment reconstruire un symbole cible à partir de la prédiction et du résidu entier ?
7. Où la RMSE intervient-elle, et où n'intervient-elle pas ?
8. Comment retrouve-t-on les 112 bits économisés ?
9. Pourquoi +0.77 % de RMSE de test et 40 % de variables en moins sont-ils compatibles ?
10. Quelle expérience montre qu'un vrai signal peut être supprimé ?
11. Quelles conclusions demanderait-on de retirer si la tolérance de test avait échoué ?
12. Pourquoi ne peut-on plus utiliser le test actuel comme une période jamais consultée ?

Réponses courtes : passé connu contre cinq rendements futurs ; annualisation des unités ;
comparaison sur les mêmes dates ; 3+4 bits ; coefficients, intercept et q ; z=m+r ;
évaluation séparée du choix MDL ; 128−4−12 ; coût prédictif mesuré séparément de la taille ;
contrôle à signal faible ; retirer l'affirmation de tolérance respectée en conservant le
résultat de simplification ; les scores ont désormais informé notre connaissance du problème.

Si tu peux expliquer ces réponses et lire les deux graphiques principaux avec tes mots,
tu maîtrises le fil scientifique nécessaire pour les sept minutes et les questions.
