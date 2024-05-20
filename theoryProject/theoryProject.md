# A note on Is RLHF More Difficult than Standard RL? A Theoretical Perspective

L'apprentissage par renforcement à partir des feedbacks humains (RLHF) représente une avancée significative en utilisant les préférences humaines comme signal d'apprentissage pour les agents d'IA. Dans cet article de blog, nous explorons les concepts du papier "Is RLHF More Difficult than Standard RL? A Theoretical Perspective" de Yuanhao Wang, Qinghua Liu, et Chi Jin. Ce papier analyse si le RLHF est plus difficile à implémenter que le RL standard en raison de la moindre quantité d'information contenue dans les préférences humaines comparées aux signaux de récompense explicites.

## Introduction

Dans cet article de blog, nous explorons les concepts présentés dans le papier "Is RLHF More Difficult than Standard RL? A Theoretical Perspective" de Yuanhao Wang, Qinghua Liu, et Chi Jin. L'apprentissage par renforcement (RL) est une technique puissante où les agents apprennent à maximiser des récompenses cumulées en interagissant avec leur environnement. Cependant, concevoir des fonctions de récompense efficaces peut être complexe voir impossible.

Pour surmonter cette limitation, l'apprentissage par renforcement à partir des feedbacks humains (RLHF) utilise les préférences humaines comme signal d'apprentissage. Cette méthode permet d'aligner plus facilement les objectifs des agents avec les valeurs humaines et rend le processus de collecte des données plus intuitif.

Le papier examine si RLHF est plus difficile à implémenter que le RL standard. Pour répondre à cette question, les auteurs proposent des approches de réduction qui convertissent les problèmes de RLHF en problèmes de RL standard. Ils introduisent notamment l'interface Préférence-vers-Récompense (P2R) et des adaptations de l'algorithme OMLE, montrant que ces méthodes peuvent résoudre efficacement une large gamme de modèles de RLHF avec des garanties théoriques robustes.

D'un point de vue personnel, bien que mes connaissances sur le RLHF soient initialement limitées, cet article m'a permis de mieux comprendre ses possibilités et ses limites. Il est compliqué de faire une analyse approfondie, mais il est évident que les méthodes proposées peuvent offrir une alternative puissante dans certains cas du RLHF. Cependant, il est important de noter que ces méthodes ne permettent pas de répondre à 100% des défis du RLHF.

![RLHF](https://hackmd.io/_uploads/B17j6H_QR.jpg)

<p style="text-align: center;">Shoggoth with Smiley Face. Courtesy of twitter.com/anthrupad</p>

## Préliminaires

### Notations et Formulation du Problème

Le papier se concentre sur les Processus de Décision de Markov épisodiques (MDP), définis par un tuple $(H, S, A, P)$ :

- **H** : Longueur de chaque épisode.
- **S** : Espace des états.
- **A** : Espace des actions.
- **P** : Fonction de probabilité de transition. Pour chaque étape $h \in [H]$ et $s, a \in S \times A$, $P_h(\cdot | s, a)$ spécifie la distribution de l'état suivant.

### Trajectoire et Politique de Markov

Une trajectoire $\tau \in (S \times A)^H$ est une séquence d'interactions avec le MDP, définie comme $(s_1, a_1, \ldots, s_H, a_H)$. Une politique de Markov $\pi = \{\pi_h : S \to \Delta A\}_{h \in [H]}$ spécifie une distribution d'actions basée sur l'état actuel à chaque étape. En revanche, une politique générale $\pi = \{\pi_h : (S \times A)^{h-1} \times S \to \Delta A\}_{h \in [H]}$ peut choisir une action en se basant sur toute l'historique jusqu'au moment $h$.

### Problème d'Optimisation

L'objectif de l'apprentissage par renforcement est de trouver une politique optimale $\pi^*$ qui maximise la récompense cumulative attendue :
$$
\pi^* = \arg \max_{\pi} \mathbb{E} \left[ \sum_{h=1}^H r(s_h, a_h) \mid \pi \right]
$$
où $r(s_h, a_h)$ est la récompense reçue après avoir pris l'action $a_h$ dans l'état $s_h$.

Dans le cadre de RLHF, au lieu de recevoir directement des signaux de récompense, l'algorithme interagit avec un MDP sans récompense et peut interroger un **oracle** de comparaison (évaluateurs humains) pour obtenir des informations sur les préférences entre différentes trajectoires.

### Hypothèses Techniques

Les hypothèses suivantes sont essentielles pour les résultats théoriques du papier :

1. **Fonction de Lien $\sigma$** :
   - La fonction de lien $\sigma$ traduit les préférences en probabilités de comparaison. Par exemple, si $\tau_1$ et $\tau_2$ sont deux trajectoires, alors la probabilité que $\tau_1$ soit préférée à $\tau_2$ est modélisée par:  $$\sigma(r^*(\tau_1) - r^*(\tau_2))$$
   - **Hypothèse** : La fonction $\sigma$ est connue et satisfait certaines propriétés de régularité, telles qu'une dérivée inférieurement bornée ($\sigma'(x) \geq \alpha > 0$).

2. **Réalisation (Realizability)** :
   - On suppose que la vraie fonction de récompense $r^*$ appartient à une classe de fonctions connue $\mathcal{R}$. Cela signifie qu'il existe un $r \in \mathcal{R}$ tel que $r = r^*$.

3. **Dimension Eluder** :
   - La dimension Eluder est une mesure de la difficulté à identifier une fonction d'une classe de fonctions donnée en fonction des erreurs possibles. Pour une classe de fonctions $\mathcal{F}$, la dimension Eluder, notée $\dim_E(\mathcal{F}, \epsilon)$, quantifie le nombre de points $\epsilon$-indépendants nécessaires pour apprendre une fonction dans $\mathcal{F}$.
    - **$\epsilon$** représente un niveau de précision ou de tolérance en termes de performance de l'algorithme.

4. **Approximation de Fonction** :
   - Il est supposé que nous connaissons une classe de fonctions de récompense $\mathcal{R}$ et que nous pouvons utiliser des approximations de fonction pour modéliser les récompenses.

### Types de Préférences

Le papier distingue deux types de préférences pour modéliser les feedbacks humains :

1. **Préférences Basées sur l'Utilité** :
   - **Définition** : Ces préférences sont modélisées par une fonction de récompense sous-jacente $r^*$. L'oracle de comparaison compare deux trajectoires $\tau_1$ et $\tau_2$ en se basant sur la différence de leurs récompenses.

2. **Préférences Générales** :
   - **Définition** : Pour des préférences plus complexes, on introduit le concept du von Neumann winner. Une politique $\pi^*$ est un von Neumann winner si elle maximise l'utilité moyenne contre toute autre politique, même lorsque les préférences ne peuvent pas être modélisées par une simple fonction d'utilité. (Le concept de Von Neumann sera détaillé dans le section dediés)

## Motivating Example

The subject is not extremely deep and complex, but an example never hurt anyone. Here’s how RLHF can be applied in a practical scenario.

### Exemple : Entraînement d'un Robot de Service à Domicile

Imaginez développer un robot de service destiné à aider les personnes âgées à domicile. La conception d'une fonction de récompense précise pour chaque tâche, comme préparer le thé ou ranger la cuisine, peut être complexe et subjective.

![robot_who_help](https://hackmd.io/_uploads/Hyx5QOOX0.jpg)


#### Défis de la Conception des Récompenses

1. **Complexité et Subjectivité des Tâches** :
   - La préparation du thé peut inclure plusieurs étapes (faire bouillir de l'eau, mettre le thé dans la théière, verser l'eau, etc.), chacune ayant ses propres critères de réussite.
   - La notion de "cuisine bien rangée" peut varier considérablement d'une personne à l'autre.

2. **Préférences Utilisateur** :
   - Les préférences personnelles jouent un rôle crucial. Par exemple, une personne peut préférer un thé plus fort, tandis qu'une autre préfère un thé plus léger.
   - Les attentes peuvent également varier en fonction du contexte.

#### Application de RLHF

1. **Collecte des Préférences** :
   - Le robot effectue différentes tentatives pour préparer le thé ou ranger la cuisine.
   - Les utilisateurs fournissent des feedbacks en comparant deux résultats de ces tentatives.

2. **Utilisation de l'Interface Préférence-vers-Récompense (P2R)** :
   - L'interface P2R convertit ces préférences en signaux de récompense approximatifs. (Nous allons detaillez cette algorithme dans la prochaine partie)

3. **Apprentissage et Adaptation** :
   - Le robot utilise ces estimations pour améliorer ses politiques et mieux aligner ses actions sur les préférences des utilisateurs.

Cet exemple montre comment RLHF peut transformer des tâches complexes et subjectives en processus d'apprentissage gérables, alignant les actions des agents sur les préférences humaines.

## Partie 3 : Préférences Basées sur l'Utilité

### Introduction

Les préférences basées sur l'utilité modélisent les préférences humaines en termes de récompenses. Plutôt que de définir directement des fonctions de récompense, l'apprentissage par renforcement à partir des feedbacks humains (RLHF) utilise les comparaisons de trajectoires pour dériver des fonctions de récompense approximatives.

### P2R: Preference to Reward

L'algorithme P2R (Preference to Reward) convertit les préférences humaines en récompenses utilisables par les algorithmes de RL standard. Il facilite l'intégration des feedbacks humains dans le processus d'apprentissage, permettant aux agents d'aligner leurs actions avec les préférences des utilisateurs.

![P2R_algo](https://hackmd.io/_uploads/r169DcvXR.png)

### Fonctionnement de P2R

1. **Ensemble de Confiance pour les Récompenses** :
   
   P2R maintient un ensemble de confiance $B_r$ contenant les fonctions de récompense possibles basées sur les préférences observées. Cet ensemble de confiance est initialisé pour inclure toutes les fonctions de récompense compatibles avec les préférences exprimées par les utilisateurs.

2. **Oracle de Comparaison** :
   
   Lorsqu'un nouvel échantillon de trajectoire $\tau$ est obtenu, P2R décide s'il est nécessaire de consulter l'oracle de comparaison pour obtenir un feedback. Si les fonctions de récompense possibles dans $B_r$ s'accordent suffisamment sur la récompense de $\tau$, P2R utilise cette estimation sans interroger l'oracle. Sinon, P2R interroge l'oracle pour comparer $\tau$ avec une trajectoire de référence $\tau_0$.

3. **Mise à Jour de l'Ensemble de Confiance** :
   
   À chaque interrogation de l'oracle, l'ensemble $B_r$ est mis à jour pour inclure uniquement les fonctions de récompense compatibles avec les nouvelles comparaisons. Ce processus de mise à jour utilise des techniques de maximum de vraisemblance pour ajuster les estimations de la fonction de récompense réelle.


### Utilité de P2R

![P2R_schema](https://hackmd.io/_uploads/ByIK4dOXA.png)


1. **Économie de Requêtes à l'Oracle** :
   - P2R minimise le nombre de requêtes nécessaires à l'oracle en utilisant des estimations de récompense lorsque possible. Cela réduit la charge de travail pour les évaluateurs humains et rend l'apprentissage plus efficace.

2. **Compatibilité avec les Algorithmes de RL Standard** :
   - En convertissant les préférences en récompenses approximatives, P2R permet l'utilisation directe des algorithmes de RL standard. Les algorithmes de RL tels que Q-learning, SARSA, ou les méthodes basées sur les politiques peuvent ainsi bénéficier des préférences humaines sans nécessiter de réingénierie significative.

3. **Robustesse Théorique** :
   - P2R fournit des garanties théoriques sur l'efficacité de l'apprentissage, assurant que les politiques apprises sont proches de l'optimalité avec un nombre d'échantillons raisonnable.

### Instanciations de P2R

1. **MDPs Tabulaires** :
   - **Définition** : Un MDP tabulaire est un type de MDP où l'espace des états et des actions est discret et de taille relativement modeste, permettant de représenter les transitions et les récompenses sous forme de tableaux.
   - **Algorithme** : Pour les MDPs tabulaires, P2R peut être utilisé avec l'algorithme UCBVI-BF (Upper Confidence Bound for Value Iteration with Bonus Function).

2. **Problèmes de RL avec Faible Dimension Bellman-Eluder** :
   - **Définition** : La dimension Bellman-Eluder mesure la complexité d'un problème de RL en termes de dépendance des états et des actions. Un problème avec une faible dimension Bellman-Eluder a une structure qui facilite l'apprentissage.
   - **Algorithme** : Pour les problèmes de RL avec une faible dimension Bellman-Eluder, P2R peut être utilisé avec l'algorithme GOLF (Gradient-Optimistic Linear Function).

3. **MDPs avec Dimension Eluder Généralisée** :
   - **Définition** : La dimension Eluder généralisée est une extension de la dimension Eluder qui s'applique à des classes de fonctions plus complexes et plus générales.
   - **Algorithme** : Pour les MDPs avec une dimension Eluder généralisée, P2R peut être utilisé avec l'algorithme OMLE (Optimistic Model-based Learning).


### Analyse Théorique de P2R

L'analyse théorique de P2R montre que cet algorithme permet d'apprendre une politique $\epsilon$-optimale en convertissant efficacement les préférences humaines en récompenses utilisables par les algorithmes de RL standard.

**Complexité d'échantillonnage** : P2R a une complexité d'échantillonnage proportionnelle à la taille de l'espace d'état et d'action, ainsi qu'à la longueur de l'épisode, assurant un apprentissage efficace même dans des environnements complexes.

**Complexité de requête** : P2R minimise les requêtes à l'oracle en utilisant des estimations de récompense lorsque possible. La complexité de requête dépend de la précision souhaitée, mais reste gérable grâce à l'utilisation d'ensembles de confiance et d'interrogations ciblées.

**Robustesse théorique** : Les garanties théoriques de P2R incluent la convergence vers des politiques $\epsilon$-optimales avec un nombre raisonnable d'échantillons et de requêtes, assurant une robustesse comparable aux algorithmes de RL standard.

P2R optimise ainsi la complexité d'échantillonnage et de requête tout en garantissant l'apprentissage de politiques $\epsilon$-optimales à partir des préférences humaines.

### P-OMLE : Optimistic Model-based Learning from Preferences

La méthode P-OMLE (Preference-based Optimistic Model-based Learning) est une adaptation de l'algorithme OMLE pour traiter directement les préférences des trajectoires. Elle vise à réduire la complexité de requête tout en maintenant l'efficacité de l'apprentissage des politiques optimales à partir de feedbacks humains.

### Algorithme OMLE : Optimistic Model-based Learning with Exploration

L'algorithme OMLE (Optimistic Model-based Learning with Exploration) provient d'un autre papier *(Optimistic MLE—A Generic Model-based Algorithm for Partially Observable Sequential Decision Making)* et n'est pas détaillé dans le papier principal que nous discutons. Voici une brève explication pour fournir du contexte.

OMLE utilise une planification optimiste et favorise l'exploration pour apprendre des politiques optimales. Il commence par définir un ensemble de confiance pour les fonctions de récompense et de transition. À chaque étape, il planifie de manière optimiste en utilisant les meilleures estimations actuelles, exécute la politique pour collecter des données, et met à jour les estimations des modèles en fonction des nouvelles observations. Ce processus se répète jusqu'à convergence.

L'algorithme est avantageux pour son exploration efficace, son adaptabilité et ses garanties théoriques robustes, assurant la convergence vers des politiques optimales avec un nombre raisonnable d'échantillons.

#### Fonctionnement de P-OMLE

![P-OMLE_algo](https://hackmd.io/_uploads/ByxUO9wQR.png)


1. **Initialisation**
   
   **Ensemble de confiance initial $B_1$** : P-OMLE commence par définir un ensemble de confiance $B_1$ pour les fonctions de récompense et de transition, initialisé pour inclure toutes les fonctions compatibles avec les préférences observées.

2. **Planification Optimiste**
   
   **Optimisme dans la face de l'incertitude** : À chaque étape $t$, P-OMLE effectue une planification optimiste pour déterminer la politique $\pi^t$ et les fonctions de récompense et de transition associées $(r^t, p^t)$ qui maximisent la valeur estimée. Cette planification utilise une approche basée sur l'optimisme, en supposant que les meilleures estimations actuelles sont correctes.

3. **Collecte de Données**
   
   **Exécution de la politique** : P-OMLE exécute la politique $\pi^t$ pour collecter une nouvelle trajectoire $\tau$. Cette trajectoire est ensuite comparée avec une trajectoire de référence $\tau_0$ en utilisant l'oracle de comparaison, qui fournit un feedback sur les préférences entre $\tau$ et $\tau_0$.

4. **Mise à Jour de l'Ensemble de Confiance**
   
   **Actualisation basée sur les préférences** : L'ensemble de confiance $B_t$ est mis à jour en fonction des nouvelles données de comparaison, utilisant des techniques de maximum de vraisemblance pour ajuster les estimations des fonctions de récompense et de transition. Chaque nouvelle comparaison fournit des informations supplémentaires sur la fonction de récompense et permet d'exclure certaines fonctions de $B_t$ qui ne sont plus compatibles avec les observations.

#### Utilité de P-OMLE

P-OMLE présente plusieurs avantages :

1. **Réduction de la Complexité de Requête**
   - **Économie de requêtes** : P-OMLE réduit le nombre de requêtes nécessaires à l'oracle en limitant les interrogations uniquement lorsque l'incertitude est élevée.

2. **Adaptabilité**
   - **Flexibilité aux préférences complexes** : P-OMLE s'adapte aux préférences des trajectoires, permettant de traiter des feedbacks humains plus complexes et variés.

3. **Robustesse Théorique**
   - **Garanties de performance** : P-OMLE est soutenu par des garanties théoriques solides.  
Concrètement, les politiques apprises convergent vers l'optimalité avec une complexité d'échantillonnage proportionnelle à la dimension généralisée Eluder du modèle, et une complexité de requête améliorée, passant d'une dépendance cubique à linéaire par rapport à $d_R$ (mesure de la complexité de la classe des fonctions de récompense)

### Instanciations de P-OMLE

Le papier propose plusieurs instanciations de P-OMLE pour différents types de MDP et classes de fonctions de récompense :


1. **MDPs Tabulaires Adversariaux**
   - **Définition** : Un MDP tabulaire adversarial est un MDP où les états et les actions sont discrets, mais où les récompenses peuvent être choisies de manière adversariale pour chaque épisode, rendant l'apprentissage plus difficile.
   - **Algorithme** : P-OMLE utilise un algorithme basé sur les méthodes de planification optimiste.
   - **Complexité d'échantillonnage** : $O(|S|^2 |A| H^3 / \epsilon^2)$.
   - **Complexité de requête** : Optimisée proportionnellement à la complexité de l'espace d'état et d'action.

2. **MDPs Linéaires Adversariaux**
   - **Définition** : Un MDP linéaire adversarial est un MDP où les transitions peuvent être modélisées par des fonctions linéaires, mais où les récompenses peuvent être choisies de manière adversariale.
   - **Algorithme** : P-OMLE utilise des méthodes de planification linéaire pour estimer les fonctions de récompense et de transition.
   - **Complexité d'échantillonnage** : $O(d H^2 K^{6/7})$.
   - **Complexité de requête** : Réduite grâce à l'utilisation de modèles linéaires pour les estimations de récompense et de transition.

### Extension à la Comparaison par K-éléments

Pour améliorer l'efficacité de P-OMLE, le papier propose une extension pour gérer les comparaisons par K-éléments, où l'oracle évalue plusieurs trajectoires simultanément.

#### Fonctionnement de la Comparaison par K-éléments

1. **Interrogation de l'Oracle**

    **Comparaison multiple** : Plutôt que de comparer deux trajectoires à la fois, l'oracle évalue un ensemble de $K$ trajectoires simultanément.

2. **Mise à Jour de l'Ensemble de Confiance**

    **Incorporation des comparaisons multiples** : Les informations obtenues à partir des comparaisons par K-éléments sont utilisées pour mettre à jour l'ensemble de confiance $B_t$.

#### Avantages de la Comparaison par K-éléments

1. **Réduction de la Complexité de Requête**

    **Économie de requêtes** : En interrogeant l'oracle avec plusieurs trajectoires simultanément, l'algorithme réduit le nombre total de requêtes nécessaires.

2. **Efficacité d'Apprentissage**

   **Enrichissement des données** : La comparaison par K-éléments permet de recueillir des informations plus riches et diversifiées à chaque requête, améliorant ainsi l'efficacité globale de l'apprentissage.

#### Théorème Associé

**Théorème 10** : La complexité de requête avec la comparaison par K-éléments est réduite par un facteur de $\min\{K, m\}$,

 où $m$ est le nombre de politiques exploratoires nécessaires. Cela signifie que le nombre de requêtes nécessaires à l'oracle diminue proportionnellement au nombre d'éléments comparés simultanément.

### Analyse Théorique de P-OMLE

L'analyse théorique de P-OMLE montre que cet algorithme permet d'apprendre une politique $\epsilon$-optimale avec une complexité d'échantillonnage et de requête réduite. En combinant une planification optimiste avec des mises à jour basées sur les préférences, P-OMLE assure une convergence efficace vers des politiques optimales tout en minimisant les requêtes à l'oracle.

**Complexité d'échantillonnage** : La méthode présente une complexité d'échantillonnage proportionnelle à la taille de l'espace d'état et d'action, ainsi qu'à la longueur de l'épisode, assurant que le nombre d'échantillons nécessaires pour atteindre une $\epsilon$-optimalité est gérable.

**Complexité de requête** : En utilisant des comparaisons par K-éléments et des mises à jour optimistes, P-OMLE réduit significativement le nombre de requêtes nécessaires à l'oracle, ce qui est crucial pour rendre l'apprentissage par feedback humain praticable dans des scénarios réels.

**Robustesse théorique** : Les garanties théoriques associées à P-OMLE assurent que cette méthode est fiable et efficace pour l'apprentissage par renforcement à partir des feedbacks humains.

### Modifications Potentielles pour UCBVI-BF et GOLF

Les algorithmes UCBVI-BF et GOLF peuvent également être adaptés pour mieux intégrer les préférences humaines. En ajustant les mécanismes de mise à jour et de planification pour tenir compte des préférences et des feedbacks humains, ces algorithmes peuvent améliorer leur efficacité et leur robustesse dans le contexte de RLHF. Cela pourrait inclure l'incorporation de techniques de réduction de la complexité de requête et l'optimisation des ensembles de confiance pour mieux gérer les préférences complexes.

### Différences entre P-OMLE et P2R

P2R et P-OMLE diffèrent principalement dans leur approche de l'apprentissage et de l'optimisation :

**P2R** : Offre une interface simple pour convertir les préférences en récompenses utilisables par les algorithmes de RL standard. Bien qu'efficace, P2R peut entraîner une complexité de requête élevée en raison de sa nature "boîte noire".  

**P-OMLE** : Utilise une modification "boîte blanche" de l'algorithme OMLE, permettant une analyse spécialisée et une réduction significative de la complexité des requêtes. P-OMLE se concentre sur la planification optimiste et la mise à jour des ensembles de confiance basés sur les préférences directement, ce qui améliore l'efficacité.

## Partie 4 : Apprentissage à partir de Préférences Générales

La section 4 du papier aborde les méthodes de réduction pour traiter des préférences générales, c'est-à-dire des préférences qui ne peuvent pas être directement modélisées par une fonction d'utilité. Les auteurs montrent comment ces préférences peuvent être abordées en les réduisant à des problèmes d'apprentissage dans des jeux de Markov factorisés et indépendants (FI-MG) ou à des MDPs adversariaux. Cette section détaille également l'utilisation de l'algorithme OMLE adapté à ces préférences.

### Réduction aux Jeux de Markov

#### Jeux de Markov Factorisés et Indépendants (FI-MG)

Un **Jeu de Markov Factorisé et Indépendant (FI-MG)** est un jeu de Markov à somme nulle (jeu où la somme totale des gains et des pertes est toujours nulle) avec les caractéristiques suivantes :

- **Espaces d'État et d'Action** : L'espace d'état $S$ est factorisé en deux sous-espaces $S^{(1)}$ et $S^{(2)}$, et l'espace d'action $A$ est factorisé en $A^{(1)}$ et $A^{(2)}$.
- **Transition Factorisée** : La transition entre états est également factorisée en deux composantes indépendantes :
   $$
   P_h(s_{h+1} | s_h, a_h) = P_h(s_{h+1}^{(1)} | s_h^{(1)}, a_h^{(1)}) \times P_h(s_{h+1}^{(2)} | s_h^{(2)}, a_h^{(2)})
   $$
  où $s_h = (s_h^{(1)}, s_h^{(2)})$ et $a_h = (a_h^{(1)}, a_h^{(2)})$.

- **Politiques Restreintes** : Les classes de politiques $\Pi^{(1)}$ et $\Pi^{(2)}$ contiennent des politiques qui mappent une trajectoire partielle à une distribution sur les actions, respectivement pour les sous-espaces $S^{(1)}$ et $S^{(2)}$.

#### Recherche du von Neumann Winner

**von Neumann Winner** : 

**Définition** : Une politique $\pi^*$ est un von Neumann winner si elle maximise l'utilité moyenne contre toute autre politique. Formellement, dans un jeu à somme nulle, une politique $\pi^*$ maximise le gain attendu par rapport à toute autre politique adverse. Cela signifie que, pour toute politique adverse $\pi'$, le gain espéré en suivant $\pi^*$ est au moins aussi grand que le gain espéré en suivant $\pi'$. En d'autres termes, $\pi^*$ garantit le meilleur résultat possible contre l'adversaire le plus défavorable.  

**Proposition 11** : Trouver un équilibre de Nash restreint dans un FI-MG revient à trouver un von Neumann winner dans le problème original de RLHF.

### Apprentissage à partir de Préférences basées sur l'État Final via les MDPs Adversariaux

Un **MDP Adversarial** est un cadre dans lequel l'algorithme interagit avec une série de MDPs ayant les mêmes transitions mais des récompenses choisies de manière adversariale pour chaque épisode.

#### Définition Formelle

**Regret** : Le regret est défini comme l'écart entre le gain attendu de l'algorithme et le meilleur gain possible avec une politique de Markov fixe :

$$
\text{Regret}_K(A) = \max_{\pi \in \Pi_{\text{Markov}}} \sum_{k=1}^K \mathbb{E}^\pi \left[ \sum_{h=1}^H r_h^k(s_h, a_h) \right] - \sum_{k=1}^K \mathbb{E}^{\pi^k} \left[ \sum_{h=1}^H r_h^k(s_h, a_h) \right]
$$
  où $K$ est le nombre d'épisodes, $\pi$ est une politique de Markov, et $r_h^k$ est la fonction de récompense pour l'épisode $k$.

#### Algorithme pour MDPs Adversariaux

**Algorithme 4** : Implémentation de l'apprentissage du von Neumann winner via MDPs adversariaux.

![Screenshot from 2024-05-20 21-42-20](https://hackmd.io/_uploads/ryDyDRu7R.png)


- **Étapes** :
  1. **Création de Copies Indépendantes** : Créer deux copies indépendantes du MDP original, chacune contrôlée par des algorithmes de MDPs adversariaux $A^{(1)}$ et $A^{(2)}$.
  2. **Récompenses Bernoulli** : Fournir des récompenses de type Bernoulli basées sur les états finaux observés ($s_H^{(1)}$ et $s_H^{(2)}$).
  3. **Mise à Jour des Politiques** : Mettre à jour les politiques en fonction des récompenses adversariales.

**Théorème 12** : Si l'algorithme de MDP adversarial $A$ a un regret sous-linéaire, cet algorithme peut trouver un von Neumann winner approximatif en utilisant une complexité d'échantillonnage et de requête efficace.

### Apprentissage à partir de Préférences basées sur la Trajectoire via OMLE

Pour les préférences générales basées sur la trajectoire, l'algorithme OMLE est adapté pour apprendre des politiques optimales dans des contextes où les préférences ne suivent pas un modèle d'utilité simple.

#### Fonctionnement de OMLE

1. **Hypothèse de Classe de Préférences** :
   - On suppose que l'apprenant dispose d'une classe de modèles de préférences $\mathcal{M}$ et d'une classe de fonctions de transition $\mathcal{P}$.

2. **Optimistic Model-based Learning** :
   - OMLE utilise une approche optimiste pour planifier et évaluer les trajectoires, en choisissant les politiques qui maximisent les récompenses espérées sous les préférences observées.

**Algorithme 5 : OMLE pour Préférences de Trajectoire**

![Screenshot from 2024-05-20 21-46-24](https://hackmd.io/_uploads/r1YT8ROXR.png)


- **Étapes** :
  1. **Initialisation** : Définir un ensemble de confiance initial pour les fonctions de récompense et de transition.
  2. **Planification Optimiste** : Choisir les fonctions de récompense et de transition qui maximisent les récompenses espérées.
  3. **Collecte de Données** : Exécuter les politiques optimistes pour collecter des données de trajectoire et des comparaisons de préférences.
  4. **Mise à Jour de l'Ensemble de Confiance** : Ajuster les estimations des fonctions de récompense et de transition.

**Théorème 13** : En utilisant OMLE, on peut apprendre un von Neumann winner approximatif avec une complexité d'échantillonnage de $O(H^2 d_P |Π_{exp}|^2 \ln |P| / \epsilon^2 + H d_R |Π_{exp}| / \epsilon)$.

### Comparaison des Méthodes

1. **FI-MG vs MDP Adversariaux** :
   - **FI-MG** : Utilisé pour modéliser des jeux de Markov factorisés et indépendants, où les transitions et les actions sont divisées en sous-espaces distincts.
   - **MDP Adversariaux** : Modélise des interactions adversariales dans des MDPs, avec des récompenses choisies de manière adversariale pour chaque épisode, permettant de trouver des von Neumann winners dans des contextes plus compétitifs.

2. **OMLE vs Algorithme pour MDPs Adversariaux** :
   - **OMLE** : Utilise une approche optimiste pour planifier et évaluer les trajectoires, particulièrement adapté pour des préférences générales basées sur la trajectoire.
   - **MDP Adversariaux** : Utilise des copies indépendantes du MDP original et des récompenses de type Bernoulli, efficace pour apprendre des politiques optimales avec un regret sous-linéaire.

## Discussions et Critiques

### Contributions Principales

1. **Réduction à l'apprentissage par renforcement standard** :
   - Les auteurs montrent comment les problèmes de RLHF peuvent être convertis en problèmes de RL standard, permettant ainsi l'utilisation des algorithmes de RL existants avec des garanties de robustesse. Cela prouve qu'il est possible d'étendre les méthodes de RLHF au RL sans trop de difficultés et d'utiliser ainsi les algorithmes traditionnels comme P2R, P-OMLE, UCBVI-BF, et GOLF avec peu de modifications. Cette approche facilite l'intégration des préférences humaines dans les processus d'apprentissage par renforcement, rendant les méthodes plus accessibles et applicables à une gamme plus large de problèmes.
   - Cela simplifie la mise en œuvre pratique du RLHF dans des environnements complexes en réutilisant les algorithmes éprouvés de RL standard.
   - La réduction permet également de tirer parti des avancées récentes en RL standard pour améliorer les performances de RLHF.

2. **Approches générales** :
   - Les méthodes proposées offrent une approche élégante et pratique pour exploiter les techniques de RL existantes dans des contextes de feedback humain, en s'adaptant à différents types de préférences, qu'elles soient basées sur l'utilité ou plus générales.

3. **Garanties Théoriques** :
   - Le papier fournit des preuves rigoureuses pour démontrer l'efficacité des approches proposées en termes de complexité d'échantillonnage et de requête. Les garanties théoriques solides assurent que les politiques apprises convergent vers l'optimalité avec une complexité d'échantillonnage et de requête gérable. Cela inclut des résultats montrant que P2R et P-OMLE peuvent apprendre des politiques $\epsilon$-optimales avec un nombre raisonnable d'échantillons et de requêtes, même dans des environnements complexes.

### Intuitions Derrière les Théorèmes et les Preuves

1. **Interface Préférence-vers-Récompense (P2R)** :
   - P2R convertit les préférences en récompenses utilisables par les algorithmes de RL standard, ce qui réduit la complexité de l'apprentissage et permet de bénéficier des avancées des algorithmes de RL traditionnels.

2. **OMLE et P-OMLE** :
   - OMLE et P-OMLE utilisent une approche optimiste pour planifier et évaluer les trajectoires, assurant que les politiques apprises sont proches de l'optimalité. Cette approche permet de maximiser l'utilisation des informations de préférence disponibles et de minimiser le nombre de requêtes à l'oracle.

### Faiblesses Potentielles

1. **Hypothèses Fortes** :
   - Certaines hypothèses, telles que la connaissance précise de la fonction de lien $\sigma$, peuvent ne pas être réalistes dans tous les contextes. En pratique, les préférences humaines peuvent être plus complexes et difficiles à modéliser précisément.

2. **Complexité de Requête** :
   - Bien que la complexité de requête soit réduite, elle peut encore être élevée pour des problèmes très complexes ou des préférences très nuancées. Cela pourrait limiter l'applicabilité des méthodes proposées dans des scénarios où les ressources en termes de requêtes sont limitées.

3. **Manque de Détails** :
   - Le papier pourrait bénéficier de plus de détails sur les théorèmes et les preuves pour clarifier certaines étapes et hypothèses. Une explication plus approfondie des concepts clés et des justifications théoriques renforcerait la compréhension des méthodes proposées.

### Réflexion Personnelle

En tant qu'étudiant international n'ayant pas d'expérience avec les articles de recherche avant mon semestre à NYCU, il est difficile de faire une analyse approfondie. Cependant, je pense que ces méthodes peuvent être une alternative puissante dans certains cas de RLHF, même si elles ne permettent pas de répondre à tous les défis du RLHF. Si j'étais l'auteur, je concevrais quelque chose de similaire, tout en cherchant à simplifier davantage les hypothèses et à explorer des méthodes pour estimer empiriquement la fonction de lien $\sigma$. Cela pourrait rendre les approches plus robustes et applicables dans une plus grande variété de contextes.

## Conclusion

### Résumé des Résultats

Le papier démontre que RLHF peut être réduit à des problèmes de RL standard, simplifiant ainsi l'intégration des préférences humaines dans les algorithmes de RL. Les contributions principales incluent :

- **Approches de Réduction** : Conversion de RLHF en problèmes de RL standard ou de jeux de Markov factorisés et indépendants.
- **Algorithmes Efficaces** : Introduction de P2R et P-OMLE.
- **Garanties Théoriques** : Preuves de convergence et de performance.

### Implications et Applications

Les méthodes proposées sont applicables à divers contextes de RL, tels que la robotique, les jeux, et le fine-tuning des modèles de langage, offrant des solutions pratiques et robustes pour intégrer les préférences humaines.

### Pour finir ...

En conclusion, les résultats de ce papier montrent que RLHF n'est pas intrinsèquement plus complexe que le RL standard, ouvrant de nouvelles perspectives pour l'application de RLHF dans divers domaines. En intégrant les préférences humaines de manière efficace, ces méthodes offrent des solutions pratiques pour aligner les actions des agents avec les valeurs et attentes humaines.

![meme_RLHF](https://hackmd.io/_uploads/rJlwrhOXR.jpg)

## References

1. Yuanhao Wang, Qinghua Liu, Chi Jin. "Is RLHF More Difficult than Standard RL? A Theoretical Perspective." 
2. Qinghua Liu, Praneeth Netrapalli, Csaba Szepesvári, Chi Jin. "Optimistic MLE—A Generic Model-based Algorithm for Partially Observable Sequential Decision Making."
3. Huyen Chip. "Reinforcement Learning from Human Feedback." [Huyen Chip Blog](https://huyenchip.com/2023/05/02/rlhf.html). May 2, 2023.

