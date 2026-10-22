# Dossier complet de soutenance — Shocks-and-WENO

> **Projet :** *Machine learning-based numerical scheme for CFD*<br>
> **Auteur :** Yasser BAOUZIL<br>
> **Encadrant :** Radouan BOUKHARFANE<br>
> **Contexte :** stage de deux mois, UM6P — College of Computing, juin–août 2026<br>
> **État du dépôt audité :** 11 octobre 2026<br>
> **Objet de ce document :** fournir une source unique pour préparer la soutenance : contexte, théorie, méthodes, code, expériences, problèmes, diagnostics, solutions, résultats, limites et questions probables du jury.

---

## Sommaire

1. Résumé à mémoriser
2. Contexte, objectifs et périmètre
3. Fondements mathématiques
4. ENO, WENO et TENO
5. Méthodes neuronales étudiées
6. Données et cibles
7. Unrolled Physics Loss
8. Architecture logicielle
9. Campagne expérimentale complète, Phases 1 à 21
10. Benchmarks et références
11. Métriques
12. Validation et tests
13. Ce qui a réellement fonctionné
14. Ce qui n'a pas fonctionné et pourquoi
15. Limites honnêtes
16. Perspectives
17. Trame de présentation orale
18. Questions probables du jury
19. Glossaire
20. Fichiers importants
21. Conclusion prête à prononcer
22. Références

---

## 1. Résumé à mémoriser

### 1.1 Le projet en une phrase

Le projet étudie si un réseau de neurones peut améliorer les schémas numériques WENO5/TENO5 utilisés pour capturer les chocs, d'abord par imitation d'un schéma classique, puis par entraînement différentiable directement contre la solution physique après plusieurs pas de temps.

### 1.2 Le problème scientifique

Les lois de conservation hyperboliques produisent des discontinuités : chocs, contacts et fronts. Un schéma d'ordre élevé est précis dans les régions lisses, mais peut créer des oscillations de Gibbs près d'une discontinuité. WENO et TENO règlent ce compromis avec des indicateurs de régularité conçus analytiquement. Le projet demande si un réseau peut prendre de meilleures décisions locales sans perdre stabilité, conservation et positivité.

### 1.3 La réponse finale

- Un réseau entraîné uniquement à **imiter** WENO5 ou TENO5 reproduit bien son professeur, mais ne possède pas de signal lui permettant de le dépasser systématiquement.
- Une **Unrolled Physics Loss**, calculée après une vraie trajectoire de simulation différentiable, supprime ce plafond conceptuel.
- L'entraînement direct à froid échoue. Il faut d'abord préentraîner par imitation, puis affiner avec la perte physique et ancrer les poids.
- Pour Euler, la rétropropagation à travers le solveur expose une instabilité réelle liée à la positivité et au conditionnement de la base propre de Roe. Un curriculum en longueur de rollout, une réparation de positivité correcte, une pénalité évaluée avant réparation, des bornes de magnitude et le clipping des gradients rendent enfin l'entraînement stable.
- Une combinaison convexe
  \[
  \Phi_{\mathrm{final}}=(1-\lambda)\Phi_{\mathrm{TENO5}}+\lambda\Phi_{\mathrm{NN}}
  \]
  avec une petite tête dédiée pour apprendre \(\lambda\), produit les vrais gains finaux.
- Résultat canonique final : gains sur l'advection scalaire, Burgers et Shu–Osher ; quasi-égalité légèrement défavorable sur Sod. Le cas Sod est identifié comme une limite du **backbone**, et non comme un simple problème de réglage de la porte de mélange.

### 1.4 Les chiffres finaux à connaître

| Benchmark canonique | TENO5 | NN-TENO5 JAX | Rapport NN/TENO5 | Verdict |
|---|---:|---:|---:|---|
| Advection, \(L_1\) | 3.4819e-2 | 3.3813e-2 | 0.9711 | gain de 2.9 % |
| Advection, \(L_2\) | 9.8960e-2 | 9.5521e-2 | 0.9653 | gain de 3.5 % |
| Burgers, \(L_1\) | 2.0697e-4 | 1.7873e-4 | 0.8636 | gain de 13.6 % |
| Burgers, \(L_2\) | 8.1670e-4 | 6.6493e-4 | 0.8142 | gain de 18.6 % |
| Sod, moyenne \(L_1\) des 3 variables | 8.2583e-3 | 8.2871e-3 | 1.0035 | perte de 0.35 %, quasi-égalité |
| Shu–Osher, moyenne \(L_1\) des 3 variables | 2.1319e-1 | 2.0926e-1 | 0.9816 | gain de 1.84 % |

Ces résultats correspondent aux meilleurs checkpoints équilibrés des Phases 18 et 19, pas au dernier essai chronologique. La Phase 21 rapproche légèrement Sod de la parité, mais dégrade le gain Shu–Osher ; elle n'est donc pas adoptée comme meilleur modèle global.

---

## 2. Contexte, objectifs et périmètre

### 2.1 Contexte du stage

Le stage part d'un dépôt associé aux travaux de Stevens et Colonius sur WENO5-NN. Le travail ne consiste donc pas seulement à écrire un solveur depuis zéro. Il suit quatre niveaux :

1. comprendre et fiabiliser le solveur existant ;
2. reproduire les résultats WENO5/WENO5-NN ;
3. étendre le principe à TENO5/NN-TENO5 ;
4. dépasser l'imitation grâce à une perte physique déroulée et à un solveur JAX différentiable.

### 2.2 Missions initiales

- Faire une revue bibliographique sur ENO, WENO, TENO, BVD et les schémas numériques augmentés par apprentissage.
- Construire un réseau capable de calculer ou modifier la fonction de pondération/sélection à l'intérieur d'un schéma WENO/TENO.
- Évaluer le réseau sur des problèmes canoniques de CFD compressible, contre une solution exacte ou une référence numérique convergée.

### 2.3 Question de recherche

> Peut-on apprendre une règle de reconstruction qui soit plus précise que TENO5 tout en conservant les propriétés indispensables d'un schéma de capture de choc : stabilité, faible oscillation, conservation et positivité ?

### 2.4 Sous-questions

- Un réseau entraîné à reproduire un schéma classique peut-il réellement le dépasser ?
- Une décision locale sur cinq points contient-elle assez d'information pour déterminer la meilleure reconstruction ?
- Peut-on rétropropager à travers plusieurs étapes d'un solveur volumes finis ?
- Comment différencier une décision dure de type garder/rejeter ?
- Comment maintenir \(\rho>0\) et \(p>0\) pendant un rollout Euler ?
- Le gain observé sur un horizon court survit-il sur un benchmark canonique complet ?
- Une tête de mélange suffit-elle, ou faut-il améliorer le réseau de reconstruction lui-même ?

### 2.5 Périmètre réel

- Problèmes 1D.
- Lois scalaires : advection linéaire et Burgers.
- Système : Euler compressible 1D, gaz parfait avec \(\gamma=1.4\).
- Schémas principaux : WENO5 et TENO5.
- Cas de test : marche périodique, sinus/multifréquence, Burgers, tube à choc de Sod et interaction choc–onde d'entropie de Shu–Osher.
- WENO3/WENO7 et TENO3/TENO7 existent dans le code, mais les versions neuronales d'ordre 3 et 7 ne constituent pas le résultat final du stage.
- Pas de 2D, pas de maillage non structuré, pas de Navier–Stokes visqueux.

### 2.6 Contributions personnelles principales

- Audit et correction du solveur volumes finis et de ses benchmarks.
- Extension TENO3/TENO5/TENO7 et wrappers NN-TENO.
- Pipeline NN-TENO5 complet : génération de labels, entraînement, calibration des seuils, comparaison scalaire et Euler.
- Génération de données exactes/oracles et audit de non-identifiabilité.
- Port JAX différentiable complet de TENO5, du solveur scalaire et du solveur Euler caractéristique.
- Implémentation de la Unrolled Physics Loss et de ses quatre termes.
- Diagnostic puis correction progressive de l'instabilité Euler.
- Introduction du mélange convexe et de la tête dédiée \(\lambda\).
- Campagne de 21 phases expérimentales avec conservation explicite des résultats négatifs.

---

## 3. Fondements mathématiques

## 3.1 Loi de conservation scalaire

Une loi de conservation 1D s'écrit

\[
\frac{\partial u}{\partial t}+\frac{\partial f(u)}{\partial x}=0.
\]

Exemples utilisés :

- advection linéaire : \(f(u)=a u\), avec vitesse constante \(a\) ;
- Burgers inviscide : \(f(u)=u^2/2\), avec vitesse caractéristique \(f'(u)=u\).

L'advection transporte le profil sans en changer la forme. Burgers est non linéaire : les caractéristiques se croisent et un choc peut se former.

## 3.2 Volumes finis

Le domaine est divisé en cellules. On stocke la moyenne

\[
\bar u_i(t)=\frac{1}{\Delta x}\int_{I_i}u(x,t)\,dx.
\]

L'intégration de la PDE sur une cellule donne

\[
\frac{d\bar u_i}{dt}
=-\frac{1}{\Delta x}
\left(\hat f_{i+1/2}-\hat f_{i-1/2}\right).
\]

La conservation est structurelle : le flux sortant d'une cellule est le flux entrant de la cellule voisine. Le problème numérique central devient la reconstruction du flux \(\hat f\) aux interfaces.

## 3.3 Condition CFL

Un schéma explicite exige un pas de temps limité par la vitesse d'onde :

\[
\mathrm{CFL}=\frac{a_{\max}\Delta t}{\Delta x}\leq C_{\max}.
\]

- Le solveur scalaire peut vérifier une vitesse fixe fournie par le flux splitting.
- Le solveur Euler recalcule à chaque pas \(a_{\max}=\max_i(|v_i|+c_i)\).
- Le dernier pas Euler est raccourci pour atteindre exactement le temps final.

## 3.4 Intégration temporelle SSPRK3

Le projet utilise le schéma Strong Stability Preserving Runge–Kutta d'ordre 3 :

\[
u^{(1)}=u^n+\Delta tL(u^n),
\]

\[
u^{(2)}=\frac34u^n+\frac14u^{(1)}+\frac14\Delta tL(u^{(1)}),
\]

\[
u^{n+1}=\frac13u^n+\frac23u^{(2)}+\frac23\Delta tL(u^{(2)}).
\]

Dans le code, le signe négatif de la divergence de flux est appliqué dans la méthode de pas de temps. Pour Euler, la validité physique est contrôlée à chaque sous-étape, car \(u^{(1)}\) et \(u^{(2)}\) sont réutilisés par le calcul de flux avant la fin du pas complet.

## 3.5 Flux splitting de Lax–Friedrichs

Le flux est séparé en deux parties orientées :

\[
f^\pm(u)=\frac12\bigl(f(u)\pm\alpha u\bigr).
\]

La partie positive est reconstruite de gauche à droite et la partie négative avec le stencil miroir. Pour Euler, \(\alpha\) est local à l'interface et vaut le maximum des rayons spectraux des deux cellules adjacentes.

## 3.6 Système d'Euler 1D

Le vecteur conservatif est

\[
U=
\begin{bmatrix}
\rho\\
\rho v\\
E
\end{bmatrix},
\qquad
F(U)=
\begin{bmatrix}
\rho v\\
\rho v^2+p\\
v(E+p)
\end{bmatrix}.
\]

La pression et la vitesse du son sont

\[
v=\frac{\rho v}{\rho},
\qquad
p=(\gamma-1)\left(E-\frac12\rho v^2\right),
\qquad
c=\sqrt{\frac{\gamma p}{\rho}}.
\]

Les trois vitesses caractéristiques sont \(v-c\), \(v\) et \(v+c\).

### Reconstruction caractéristique

Reconstruire séparément \(\rho\), \(\rho v\) et \(E\) peut créer des oscillations croisées. Le solveur calcule donc une moyenne de Roe à chaque interface :

1. calcul des états gauche et droit ;
2. calcul de la vitesse et de l'enthalpie moyennes de Roe ;
3. construction de la matrice propre droite \(R\) ;
4. inversion \(L=R^{-1}\) ;
5. projection des stencils conservatifs et des flux : \(W=L U\) ;
6. reconstruction indépendante des trois champs caractéristiques ;
7. retour au flux conservatif avec \(R\).

La même base propre d'interface est utilisée pour les flux positifs et négatifs. Cette dépendance de la base à l'état courant explique pourquoi une petite erreur neuronale peut devenir beaucoup plus dangereuse pour Euler que pour l'advection scalaire.

## 3.7 Conditions aux limites

- Scalaire : périodique uniquement dans le solveur principal.
- Euler : transmissive par défaut, périodique disponible.
- Limite transmissive : extrapolation constante des cellules de bord dans les cellules fantômes.
- Limite périodique : indices ramenés modulo le nombre de cellules.

---

## 4. ENO, WENO et TENO

## 4.1 Stencil WENO5/TENO5

Pour l'interface \(i+1/2\), le stencil de cinq points est

\[
[f_{i-2},f_{i-1},f_i,f_{i+1},f_{i+2}].
\]

Il contient trois sous-stencils de trois points. Les candidats d'ordre 3 sont

\[
q_0=\frac13f_{i-2}-\frac76f_{i-1}+\frac{11}{6}f_i,
\]

\[
q_1=-\frac16f_{i-1}+\frac56f_i+\frac13f_{i+1},
\]

\[
q_2=\frac13f_i+\frac56f_{i+1}-\frac16f_{i+2}.
\]

Les poids linéaires optimaux sont

\[
d=[d_0,d_1,d_2]=\left[\frac1{10},\frac35,\frac3{10}\right].
\]

## 4.2 ENO

ENO garde un seul sous-stencil, choisi comme le plus lisse. Il évite qu'un polynôme traverse directement un choc, mais la sélection dure peut changer brutalement pour une faible variation des données.

## 4.3 WENO5-JS

WENO combine tous les candidats avec des poids non linéaires :

\[
\alpha_k=\frac{d_k}{(\varepsilon+\beta_k)^2},
\qquad
\omega_k=\frac{\alpha_k}{\sum_l\alpha_l},
\qquad
\hat f_{i+1/2}=\sum_k\omega_kq_k.
\]

Les indicateurs de Jiang–Shu sont

\[
\beta_0=\frac{13}{12}(f_{i-2}-2f_{i-1}+f_i)^2
+\frac14(f_{i-2}-4f_{i-1}+3f_i)^2,
\]

\[
\beta_1=\frac{13}{12}(f_{i-1}-2f_i+f_{i+1})^2
+\frac14(f_{i-1}-f_{i+1})^2,
\]

\[
\beta_2=\frac{13}{12}(f_i-2f_{i+1}+f_{i+2})^2
+\frac14(3f_i-4f_{i+1}+f_{i+2})^2.
\]

Dans une région lisse, \(\omega_k\to d_k\) et l'ordre 5 est récupéré. Près d'un choc, le poids du candidat traversant la discontinuité devient très faible.

## 4.4 TENO5

TENO utilise les mêmes candidats, mais effectue une séparation plus nette :

\[
\tau_5=|\beta_0-\beta_2|,
\]

\[
\gamma_k=\left(C+\frac{\tau_5}{\beta_k+\varepsilon}\right)^q,
\qquad
\chi_k=\frac{\gamma_k}{\sum_l\gamma_l}.
\]

La décision est

\[
\delta_k=
\begin{cases}
1,&\chi_k\ge C_T,\\
0,&\chi_k<C_T.
\end{cases}
\]

Puis

\[
\omega_k=\frac{d_k\delta_k}{\sum_l d_l\delta_l},
\qquad
\hat f=\sum_k\omega_kq_k.
\]

Paramètres exacts du dépôt :

| Paramètre | Valeur |
|---|---:|
| \(\varepsilon\) | \(10^{-40}\) |
| \(C\) | 1 |
| \(q\) | 6 |
| \(C_T\) | \(10^{-5}\) |

La version NumPy calcule \(\gamma_k\) dans le domaine logarithmique puis applique une normalisation de type softmax afin d'éviter les dépassements numériques. Si tous les candidats sont rejetés, elle retombe sur la valeur centrale/upwind, donc sur un comportement de premier ordre défini.

## 4.5 Différence conceptuelle WENO/TENO

- WENO réduit continûment le poids d'un mauvais candidat.
- TENO décide presque binairement de le conserver ou de l'éliminer.
- TENO garde exactement les poids optimaux sur les candidats retenus, ce qui améliore la résolution spectrale en zone lisse.
- La décision discrète TENO est précisément l'élément que NN-TENO cherche à apprendre.

---

## 5. Méthodes neuronales étudiées

## 5.1 WENO5-NN : correction de coefficients

Le réseau historique ne prédit pas directement une solution physique arbitraire. Il part des coefficients WENO classiques, calcule une correction bornée par \(0.5\tanh(\cdot)\), puis applique une correction affine non entraînable assurant que la somme des coefficients reste égale à 1. La reconstruction finale est le produit scalaire entre le stencil et les coefficients corrigés.

Cette architecture injecte un biais utile : le réseau commence près d'un schéma connu et conserve la consistance sur les constantes.

La normalisation par stencil est

\[
x'_j=\frac{x_j-\min(x)}{\max(x)-\min(x)}.
\]

Un stencil constant devient zéro pour l'entrée du réseau et sa valeur centrale est directement restituée.

## 5.2 NN-TENO5 Keras : classifieur de candidats

NN-TENO5 ne prédit ni le flux final ni les polynômes candidats. Il prédit trois probabilités indépendantes :

- 0 : candidat lisse, à garder ;
- 1 : candidat troublé, à rejeter.

Architecture :

```text
5 entrées -> Dense 64 ReLU -> Dense 32 ReLU
           -> Dense 16 ReLU -> Dense 8 ReLU
           -> 3 sorties sigmoid
```

Entraînement : Adam, binary cross-entropy, labels multi-étiquettes. Le modèle final d'imitation utilise des seuils calibrés par candidat :

```text
[0.66, 0.50, 0.69]
```

Le jeu original contient 75 241 stencils. La séparation est stratifiée 70/15/15 sur les huit masques possibles. Le modèle enregistré obtient :

- binary accuracy : 0.9954 ;
- exact-mask accuracy : 0.9869 ;
- macro F1 : 0.9911.

Ces scores montrent une excellente imitation de TENO5, mais pas une amélioration physique garantie.

## 5.3 Modèle JAX différentiable

Le backbone JAX reprend l'architecture réelle 5–64–32–16–8–3. Les couches cachées utilisent ReLU et la sortie sigmoid. Les poids sont initialisés selon He pour les couches ReLU.

La fonction `toy_keep_weight` :

1. normalise chaque stencil par min–max ;
2. protège le dénominateur sur les stencils constants ;
3. produit une probabilité *troubled* ;
4. renvoie \(1-p_{\mathrm{troubled}}\), donc un poids continu de conservation.

La protection du dénominateur est importante en autodiff : un `where` qui masque seulement le résultat ne suffit pas, car la branche non choisie peut tout de même produire un gradient \(1/0\) et contaminer le backward pass.

## 5.4 Projection soft/hard

- En entraînement : \(w_k^{\mathrm{keep}}=1-p_k\), continu et différentiable.
- En inférence : \(w_k^{\mathrm{keep}}=\mathbf{1}[p_k<\theta_k]\), décision dure identique au NN-TENO classique.

La combinaison des candidats renormalise toujours les poids actifs. Si leur somme est nulle, le fallback central évite toute division indéfinie.

## 5.5 Mélange convexe avec TENO5

La reconstruction finale est

\[
\Phi_{\mathrm{mix}}=(1-\lambda)\Phi_{\mathrm{TENO5}}
+\lambda\Phi_{\mathrm{NN}},
\qquad 0\le\lambda\le1.
\]

Trois variantes ont été testées :

1. \(\lambda=\max_k p_k\), sans nouveaux paramètres ;
2. une tête dédiée linéaire 3→1 prenant les trois probabilités comme entrée ;
3. une tête enrichie prenant 3 probabilités + 5 valeurs du stencil normalisé, avec une couche cachée.

La meilleure solution globale est la tête dédiée simple. Elle est initialisée avec des poids presque nuls et un biais \(-3\), soit \(\sigma(-3)\approx0.047\). Elle commence donc très près de TENO5, puis ne s'en écarte que si la perte physique le justifie.

---

## 6. Données et cibles

## 6.1 Données historiques WENO5

- `data/2ndNewAvgs.csv` : matrice 5 × 75 241 avant transposition logique en échantillons.
- `data/2ndNewFlux.csv` : une cible de reconstruction par stencil.
- Valeurs d'entrée : environ [0, 1].
- Cibles : environ [-0.5, 1.5].
- Les entrées et cibles utilisent le minimum et l'étendue du stencil pour la normalisation.

## 6.2 Labels NN-TENO5 par imitation

Les labels ne proviennent pas du fichier de flux. Ils sont recalculés à partir de chaque stencil par la règle TENO5 déterministe : trois bits correspondant aux trois candidats.

Avantage : labels cohérents, faciles à apprendre et schéma stable.

Limite : le réseau apprend la décision de TENO5, pas l'erreur contre la vraie solution.

## 6.3 Données Euler exactes

Le projet construit des problèmes de Riemann aléatoires et des trajectoires choc–oscillation. Les cibles peuvent venir :

- de la solution exacte du problème de Riemann ;
- d'une référence fine WENO7/TENO5 ;
- d'une valeur d'interface exacte lorsque la famille analytique le permet.

Les données exactes existent en fenêtres de 5 et 7 points. Les configurations susceptibles de produire un vide sont rejetées. Les amplitudes extrêmes sont bornées afin d'éviter qu'une poignée d'outliers domine la MSE.

## 6.4 Dataset oracle TENO5

`GenerateTrainingData_TENOExact.py` génère 75 000 stencils issus de cinq familles :

| Famille | Part | Désaccord du masque oracle avec TENO5 |
|---|---:|---:|
| sinus | 30 % | 24.3 % |
| double sinus | 20 % | 40.2 % |
| polynôme cubique | 20 % | 1.4 % |
| marche | 20 % | 28.6 % |
| kink \(|x|\)-like | 10 % | 18.9 % |
| total | 100 % | 23.2 % |

Pour chaque stencil, les huit masques garder/rejeter possibles sont testés. L'oracle choisit celui dont la reconstruction est la plus proche de la valeur exacte à l'interface.

### Bug de tie-breaking découvert

Sur un stencil plat, plusieurs masques donnent exactement zéro erreur. `itertools.product` énumérait d'abord « tout rejeter », ce qui créait de faux labels troublés. La correction a été :

- tester « tout garder » en premier ;
- n'accepter une amélioration que si elle dépasse une tolérance de \(10^{-10}\).

## 6.5 Non-identifiabilité locale

Pour Shu–Osher, des stencils 5 points presque identiques, séparés de seulement 0.015–0.028 en norme \(L_2\), peuvent avoir des cibles exactes distantes de 2.3 à 4.5. La bonne réponse dépend de l'histoire amont des ondes, invisible dans la fenêtre locale.

Conclusion : ce n'est pas toujours un problème de capacité réseau. Une fonction déterministe locale `5 points -> cible exacte` peut ne pas exister. Le passage à 7 points sépare fortement la majorité de ces paires, mais ne résout pas à lui seul toute la dynamique.

## 6.6 Vecteur de caractéristiques proposé

Le vecteur étendu comprend :

\[
X_j=[u_s,\tilde c,\eta_0,\eta_1,\eta_2,J_{j-1/2},J_{j+1/2},M_j,|\Delta p|/p].
\]

- \(u_s\) : stencil normalisé ;
- \(\tilde c\) : poids linéaires optimaux \([1/10,3/5,3/10]\) ;
- \(\eta_k=\tau_5/(\beta_k+\varepsilon)\) ;
- \(J\) : variations de bord BVD ;
- \(M\) et gradient de pression : caractéristiques Euler.

Une fenêtre brute de six points est nécessaire pour calculer correctement les deux interfaces d'une même cellule à partir d'échantillons isolés. Dans le solveur complet, la BVD correcte utilise directement les reconstructions des cellules voisines.

---

## 7. Unrolled Physics Loss

## 7.1 Motivation

La perte d'imitation minimise

\[
\mathcal L_{\mathrm{imit}}=D(\mathrm{NN}(x),\mathrm{TENO5}(x)).
\]

Elle ne contient jamais la vraie solution. La nouvelle idée consiste à faire évoluer le système pendant \(K\) pas avec le réseau dans la boucle, puis à comparer l'état obtenu à la vérité :

\[
U^{n+K}_{\mathrm{NN}}=\mathcal S_{\theta}^{K}(U^n),
\qquad
\mathcal L= D(U^{n+K}_{\mathrm{NN}},U^{n+K}_{\mathrm{exact}})+\cdots
\]

Le gradient traverse chaque reconstruction, chaque flux et chaque sous-étape SSPRK3.

## 7.2 Fonction de perte complète

\[
\mathcal L_{\mathrm{total}}
=\lambda_1\mathcal L_{\mathrm{MSE}}
+\lambda_2\mathcal L_{\mathrm{TV}}
+\lambda_3\mathcal L_{\mathrm{BVD}}
+\lambda_4\mathcal L_{\mathrm{Reg}}
+\lambda_5\mathcal L_{\mathrm{positivity}}.
\]

Les scripts de base utilisent :

```text
lambda_mse        = 1.0
lambda_tv         = 0.1
lambda_bvd        = 0.1
lambda_reg        = 0.1
lambda_positivity = 1.0  # Euler seulement
```

### MSE

\[
\mathcal L_{\mathrm{MSE}}=rac1N\sum_i
\left(U_i^{n+K}-U_{i,\mathrm{exact}}^{n+K}\right)^2.
\]

Elle donne le signal principal d'exactitude physique.

### Total Variation

\[
TV(U)=\sum_i|U_{i+1}-U_i|,
\]

\[
\mathcal L_{\mathrm{TV}}=max(0,TV(U^{n+K})-TV(U^n)).
\]

C'est une pénalité TVD souple : elle décourage la création d'oscillations mais ne garantit pas mathématiquement une TV non croissante.

### BVD

La définition correcte est

\[
BV(\Phi)_{i+1/2}
=|\Phi_i(x_{i+1/2})-\Phi_{i+1}(x_{i+1/2})|.
\]

La perte utilisée est

\[
\mathcal L_{\mathrm{BVD}}
=\mathrm{mean}\left(\max(0,BV_{\mathrm{NN}}-BV_{\mathrm{TENO5}})\right).
\]

Le réseau n'est donc pénalisé que lorsqu'il crée plus de désaccord inter-cellules que TENO5 à la même interface.

### Régularisation/ancrage

\[
\mathcal L_{\mathrm{Reg}}=\|\theta-\theta_{\mathrm{anchor}}\|_2^2.
\]

Après préentraînement d'imitation, l'ancre est le checkpoint stable obtenu. Ce terme empêche la fine-tune physique de détruire immédiatement un comportement déjà raisonnable.

### Positivité Euler

La pénalité mesure les profondeurs de violation de densité et pression avant réparation. Elle est bornée par cellule pour qu'un état astronomique ne rende pas toute la moyenne infinie.

## 7.3 Normalisation des termes

Premier problème : TV est une somme sur les cellules et Reg une somme sur les paramètres, tandis que MSE est une moyenne. Avec 64 cellules, la TV brute vaut environ 0.6–0.9 contre 0.01–0.02 pour la MSE ; même pondérée par 0.1, elle dominait le gradient d'environ 4×.

Solution :

- diviser la variation de TV par le nombre de différences intérieures ;
- diviser Reg par le nombre de paramètres.

Cette correction rétablit une échelle comparable, mais ne suffit pas sans préentraînement.

## 7.4 Préentraînement puis fine-tuning

L'entraînement à froid reçoit un signal tardif après \(K\) étapes. Le paysage est mal conditionné et un réseau aléatoire peut déstabiliser la trajectoire avant d'apprendre.

Procédure retenue :

1. préentraînement par imitation TENO5 sur stencils ;
2. utilisation de ce checkpoint comme initialisation ;
3. utilisation du même checkpoint comme ancre de \(L_{\mathrm{Reg}}\) ;
4. fine-tuning avec la vraie perte physique déroulée.

## 7.5 Rollout JAX

Une boucle Python naïve dans une fonction `jax.jit` déroule physiquement les \(K\) copies du flux dans le graphe XLA. Sur Euler et 240 sous-étapes, la compilation a pris environ 647 s.

Le remplacement par `jax.lax.scan` compile une seule fois le corps de boucle : environ 4.4 s, soit près de 150× plus rapide à compiler, pour le même résultat et environ 0.15 s par appel après compilation.

---

## 8. Architecture logicielle

```text
data/                     données d'entraînement et labels oracle
models/                   modèles, checkpoints, historiques, JSON de métriques
plots/                    figures finales et script de régénération
references/               articles et proposition de perte physique
scripts/
  legacy_imitation_training/
                            pipeline NumPy/Keras historique
  unrolled_physics_loss/    pipeline JAX différentiable
src/
  core/                    volumes finis, pas de temps, flux, Euler
  schemes/                 ENO/WENO/TENO et wrappers neuronaux
  networks/                architectures Keras et chargement de données
  differentiable/          ports JAX et loss physique
  data/                    preprocessing et feature vector
  analysis/                convergence et références Euler
  initial_conditions/      conditions initiales
  viz/                     métriques et visualisation
tests/                     tests numériques, TENO, features et régressions JAX
Documentation/             rapports et journal complet des expériences
```

## 8.1 Modules NumPy/Keras importants

| Module | Rôle |
|---|---|
| `src/core/SimulationClasses.py` | maillage, boucle temporelle, stencils, frontières, validation Euler |
| `src/core/TimeSteppingMethods.py` | Euler explicite, SSPRK2, SSPRK3 |
| `src/core/FluxSplittingMethods.py` | Lax–Friedrichs scalaire |
| `src/core/eulerEquations.py` | flux Euler, vitesses d'onde, base de Roe, flux caractéristique |
| `src/schemes/weno5.py` | WENO5 scalaire/Euler et wrappers WENO5-NN |
| `src/schemes/teno5.py` | TENO5 robuste, labels et wrappers NN-TENO5 |
| `src/networks/WenoNetworks.py` | réseaux correcteurs WENO3/5/7 |
| `src/networks/TenoNetworks.py` | classifieurs NN-TENO3/5/7 |

## 8.2 Modules JAX importants

| Module | Rôle |
|---|---|
| `src/differentiable/teno5_jax.py` | candidats, indicateurs, TENO dur et combinaison soft |
| `src/differentiable/numerics_jax.py` | stencils périodiques, flux scalaire, SSPRK3, `lax.scan`, BVD |
| `src/differentiable/euler_jax.py` | Euler caractéristique, Roe, flux, pénalité et réparation de positivité |
| `src/differentiable/projection_jax.py` | passage soft/hard et mélange convexe |
| `src/differentiable/losses_jax.py` | cinq composantes de la perte |
| `src/differentiable/teno5_jax_toy.py` | backbone MLP et têtes de mélange |

## 8.3 Pourquoi deux environnements

La stack historique impose Python 3.7, TensorFlow 1.15 et Keras 2.2.4. JAX exige une stack moderne. Les mélanger modifierait ou casserait la reproductibilité.

| Environnement | Usage | Python |
|---|---|---:|
| `weno-nn` | Keras historique, entraînement/imitation | 3.7 |
| `teno-jax` | solveur différentiable et perte physique | 3.11 |
| numérique seul | solveur déterministe et tests sans NN | moderne |

---

## 9. Campagne expérimentale complète : problème → diagnostic → solution → résultat

## Phase 1 — baseline WENO5/WENO5-NN

**But :** reproduire la littérature avant toute extension.

- Marche périodique : avantage clair du NN ; le profil reste plus net et la TV plus proche de 2.
- Sod : quasi-égalité, avec de petites différences de largeur et TV.
- Burgers : résultat mixte selon la résolution ; 3 grilles sur 4 favorisent légèrement le NN.
- Shu–Osher : légère perte dans le journal de résultats principal, avec TV plus élevée dans la région oscillatoire.

La formulation doit rester précise : sous les réglages de l'article, la densité \(L_1\) peut favoriser WENO5-NN, tandis que le maximum absolu global et la TV peuvent être légèrement défavorables. Ces métriques ne mesurent pas la même propriété ; il ne faut donc pas résumer Shu–Osher par un unique chiffre sans nommer la métrique.

**Leçon :** le gain est dépendant du problème, de la résolution et de la métrique. Une victoire sur une marche simple ne démontre pas une supériorité universelle.

## Phase 2 — cibles Euler exactes sur cinq points

**Problème :** l'imitation de WENO ne peut pas apprendre une correction vers la vérité.

**Solution tentée :** générer des cibles exactes avec solveur de Riemann et références fines Shu–Osher.

**Résultat :** échec. Même sur les échantillons difficiles, le réseau ne bat pas WENO de façon fiable.

**Diagnostic :** non-identifiabilité du mapping local cinq points → cible exacte. Des entrées presque identiques possèdent des cibles très différentes à cause de l'histoire des ondes.

## Phase 3 — stencil de sept points

**Hypothèse :** un contexte plus large doit lever l'ambiguïté.

**Corrections réalisées :**

- filtrage d'outliers jusqu'à 562 qui dominaient la MSE ;
- correction de `eps` dans WENO7-NN : \(10^{-6}\) faisait sous-déborder des termes float32 et créait `0/0 = NaN`; \(10^{-4}\) stabilise les données réelles et les stress tests.

**Résultat :** l'entraînement devient stable, mais le modèle reste moins précis et plus oscillatoire que WENO5.

## Phase 4 — recherche d'hyperparamètres WENO7-NN

**Action :** environ 140 tentatives, largeur cachée portée jusqu'à 32, recherche de régularisation et taux d'apprentissage.

**Meilleur résultat :** 26 critères sur 28 réussis ; gain de 2.4 % sur l'erreur \(L_1\) régionale Shu–Osher, mais échec sur la TV de Sod et le \(L_\infty\) régional Shu–Osher.

**Conclusion :** near-pass, pas modèle accepté.

## Phase 4b — fine-tuning ancré

**Problème :** une fine-tune libre détruit le bon compromis du checkpoint.

**Solution :** pénalité L2-SP vers les poids du near-pass.

**Résultat :** les critères principaux deviennent reproductibles, mais deux critères étroits restent incompatibles. La technique d'ancrage est validée, le modèle global non accepté.

## Phase 5 — audit complet NN-TENO5 imitation

**Constat :** pipeline 5–64–32–16–8–3 sain, entraînement reproductible, exact-mask accuracy environ 98.7 %.

**Résultats :** six benchmarks non triviaux passent la fidélité au professeur. Trois cas scalaires sont identiques bit à bit à TENO5 ; les différences sur Sod/Shu–Osher restent faibles.

**Conclusion :** excellente fidélité, pas encore preuve de supériorité.

## Phase 6 — critère `beats_teacher`

**Changement méthodologique :** déclarer un gain uniquement si l'erreur du NN contre la vraie référence est inférieure à celle de TENO5 contre cette même référence.

**Résultat imitation :** 5/6 si l'on compte les égalités ; seulement deux petits gains réels, trois égalités exactes et une perte. Cela ressemble à du bruit autour de la parité, conformément à l'objectif d'imitation.

## Phase 7 — labels oracle TENO5

**Action :** recherche exhaustive des huit masques contre une valeur exacte sur cinq familles synthétiques.

**Résultat :** 23.2 % de désaccord avec TENO5, donc un signal non trivial existe. Le bug de tie-breaking « tout rejeter » est découvert et corrigé.

## Phase 8 — entraînement oracle pur

**Résultat négatif majeur :**

- Sod s'arrête avec densité non positive au pas 6 ;
- marche d'advection : erreur \(L_1=5.34\times10^9\), soit environ \(10^{11}\) fois pire ;
- Burgers : quasi-égalité légèrement défavorable ;
- cas lisses : inchangés.

**Diagnostic :** l'optimalité locale one-shot ne préserve pas la stabilité d'une trajectoire récurrente.

## Phase 9 — oracle avec warm start et ancrage

**Solution :** démarrer du modèle d'imitation stable et pénaliser l'écart aux poids initiaux.

**Résultat :** toute la gamme d'ancrages 1–500 supprime les blow-ups. Mais aucun réglage ne dépasse le baseline d'imitation sur l'ensemble des six cas. Un faible ancrage améliore Burgers mais perd Euler ; un fort ancrage récupère Sod mais jamais Shu–Osher.

**Leçon :** l'ancrage restaure la stabilité, pas automatiquement la généralisation.

## Phase 10 — première Unrolled Physics Loss

**Infrastructure créée :** port JAX complet, projection soft, perte physique, features et tests float64.

**Scalaire :**

- pertes brutes : 0/3 ;
- pertes normalisées : 0/3 ;
- préentraînement imitation + ancre : 2/3, y compris à 3× l'horizon appris.

**Validation canonique initiale :** perte sur la marche longue et Burgers OOD. Le résultat positif existe, mais est étroit.

**Euler :** NaN dès le pas d'entraînement 0. Un état raide atteint le floor de densité, puis \(5.7\times10^{144}\). Augmenter arbitrairement la dissipation n'aide pas : 1.5× retarde, 2–5× accélère l'explosion. Le problème n'est donc pas simplement un \(\alpha\) de Lax–Friedrichs trop petit.

## Phase 11 — curriculum en longueur de rollout

**Solution issue de la littérature :** entraîner pour \(K=2\), puis 4, 8 et 15 en conservant l'état Adam.

**Résultat :** K=2 complet ; divergence à l'étape 39/60 de K=4. Amélioration mesurable, solution incomplète.

## Phase 12 — clarification BVD et positivité

**Corrections :**

- BVD compare réellement deux cellules voisines, pas un stencil à son miroir ;
- violation isolée : moyenne des deux voisins conservatifs ;
- cluster de violations : considéré comme instabilité ;
- fallback numérique pour garder `lax.scan` fini.

**Régression découverte :** un floor \(10^{-8}\) rend \(p/\rho\), donc la vitesse du son, gigantesque. Correction à \(\varepsilon=0.05\) et vitesse plafonnée.

**Résultat :** K=2 et K=4 complets ; divergence au pas 6/60 de K=8.

**Nouveau diagnostic :** la pénalité de positivité reste toujours nulle car elle observe l'état après réparation.

## Phase 13 — pénalité de positivité réellement active

**Solution :** accumuler la pénalité sur les trois sous-étapes SSPRK3 avant réparation dans un rollout spécialisé.

**Deux autres défauts corrigés :**

- `NaN <= 0` est faux, donc les NaN n'étaient pas marqués en violation ;
- un état énorme mais positif n'était pas marqué non plus.

Le masque inclut maintenant non-finitude, négativité et magnitude > \(10^3\). La pénalité par cellule est aussi plafonnée à \(10^3\).

**Résultat :** curriculum K=2→4→8→15 entièrement terminé pour la première fois. Les tests courts donnent 3/3 gains, mais les benchmarks complets donnent 1.29× l'erreur TENO5 sur Sod et 1.24× sur Shu–Osher. L'erreur est surtout dans les régions lisses.

## Phase 14 — mélange de données lisses Euler

**Action :** ajout d'ondes d'entropie lisses à vitesse et pression uniformes, avec cible analytique.

**Premier essai :** pic de loss \(1.5\times10^{10}\) au dernier pas et checkpoint écrasé.

**Problème de processus :** absence de sauvegarde du checkpoint validé.

**Solution :** reproduction et sauvegarde du checkpoint Phase 13 ; ajout de `clip_by_global_norm(1.0)` avant Adam ; conservation du meilleur état par validation.

**Résultat :** Sod passe de 1.29× à 1.16× l'erreur TENO5 ; Shu–Osher de 1.24× à 1.23×. Amélioration réelle, pas encore un gain.

## Phase 15 — curriculum scalaire élargi

**Action :** K=15→40→80→128 et ajout de familles Burgers exactes avant choc.

**Résultat négatif propre :** entraînement stable, mais advection 1.42×→1.45× et Burgers 3.62×→3.95×. L'élargissement d'horizon et de physique ne suffit pas.

## Phase 16 — mélange convexe à l'inférence

**Idée :** revenir vers TENO5 partout où le NN est peu fiable.

**Première version sans paramètre :** \(\lambda=\max p_{\mathrm{troubled}}\).

**Résultat :** amélioration immédiate de tous les checkpoints, sans réentraînement :

- advection 1.42×→1.39× ;
- Burgers 3.62×→3.25× ;
- Sod 1.16×→1.15× ;
- Shu–Osher 1.23×→1.065×.

## Phase 17 — entraînement à travers le mélange

Trois tentatives indépendantes ne dépassent pas le mélange gratuit de Phase 16. Une tentative Euler est instable ; une autre est équivalente ; la tentative scalaire se dégrade.

**Diagnostic :** la même sortie `troubled_probability` sert à la fois à sélectionner les candidats et à décider combien faire confiance au réseau. Ces deux fonctions créent des gradients concurrents.

## Phase 18 — tête de mélange dédiée Euler

**Solution :** tête séparée 3→1, initialisée avec \(b=-3\).

**Résultat canonique :**

- Sod : 1.003×, perte de 0.3 %, quasi-égalité ;
- Shu–Osher : 0.982×, gain de 1.8 %.

C'est le premier gain canonique net de NN-TENO5euler.

## Phase 19 — tête dédiée scalaire

**Surprise :** avant entraînement, la tête presque uniforme gagne déjà : advection 0.9885×, Burgers 0.8546×. Concentrer le NN uniquement aux endroits « troublés » lui faisait confiance précisément là où il est le plus fragile.

**Après 200 pas :**

- advection 0.9653× en \(L_2\) ;
- Burgers 0.8142× en \(L_2\).

Le biais passe de -3.000 à -2.792, donc le mélange reste faible, environ 5.8 %.

## Phase 20 — limite structurelle de Sod

Une learning rate 50× supérieure pour la tête et 400 étapes ne gagnent pas. Un sweep fixe de \(\lambda\) montre :

| \(\lambda\) | ratio Sod | ratio Shu–Osher |
|---:|---:|---:|
| 0.00 | 1.0000 | 1.0000 |
| 0.02 | 1.0036 | 0.9899 |
| 0.047 | 1.0067 | 0.9807 |
| 0.10 | 1.0127 | 0.9619 |
| 0.20 | 1.0333 | 0.9300 |
| 0.30 | 1.0472 | 0.9313 |

Toute quantité du backbone NN dégrade Sod, alors qu'elle améliore Shu–Osher jusqu'à environ 0.2. Le plafond est dans la reconstruction du backbone sur Sod.

## Phase 21 — porte enrichie

**Action :** ajouter le stencil normalisé aux probabilités, soit huit features et une couche cachée.

**Bug corrigé :** le premier code supposait à tort un axe « candidats » dans le stencil. Or les candidats sont générés à partir du stencil ; ils ne constituent pas un axe d'entrée.

**Résultat :** Sod 1.0026×, légèrement meilleur ; Shu–Osher 0.9893×, moins bon que Phase 18. Ce compromis latéral confirme qu'une porte plus complexe ne corrige pas un backbone inférieur sur Sod.

---

## 10. Benchmarks et références

## 10.1 Advection périodique

- Équation : \(u_t+a u_x=0\).
- Référence exacte : translation périodique du profil initial.
- Teste diffusion cumulative, largeur du front, TV et conservation.
- Un bon schéma doit garder une marche nette après de nombreux passages dans le domaine.

## 10.2 Burgers

- Équation : \(u_t+(u^2/2)_x=0\).
- Profil lisse pouvant raidir et former un choc.
- Référence analytique par caractéristiques avant choc ou référence numérique fine après choc selon le script.
- Teste la généralisation à un flux non linéaire.

## 10.3 Tube à choc de Sod

États initiaux :

```text
gauche : rho=1,     u=0, p=1
droite : rho=0.125, u=0, p=0.1
discontinuité : x=0.5
```

La solution contient une raréfaction, un contact et un choc. Une solution de Riemann exacte est disponible. Ce cas vérifie simultanément position des ondes, positivité et diffusion du contact.

## 10.4 Shu–Osher

État gauche fort :

```text
rho=3.857143, u=2.629369, p=10.3333
```

État droit :

```text
rho=1+0.2*sin(5*x), u=0, p=1
```

Le choc démarre à \(x=1\). Il n'existe pas de solution fermée ; une simulation fine sert de référence. Ce benchmark teste l'interaction choc–onde d'entropie et la capacité à préserver de petites oscillations physiques sans créer d'oscillations numériques.

Une ancienne phase incorrecte `sin(5*(x-5))` a été retirée afin de suivre exactement la définition publiée.

---

## 11. Métriques

Pour \(e_i=u_i-u_i^{\mathrm{ref}}\) :

\[
L_1=\frac1N\sum_i|e_i|,
\qquad
L_2=\sqrt{\frac1N\sum_i e_i^2},
\qquad
L_\infty=\max_i|e_i|.
\]

Pour Euler, les erreurs de densité, quantité de mouvement et énergie sont calculées séparément ; le résultat principal est leur moyenne.

Autres métriques :

- total variation ;
- overshoot/undershoot ;
- minimum de densité et de pression ;
- erreur de conservation ;
- largeur du choc ;
- position du choc et du contact ;
- erreur régionale dans la zone oscillatoire Shu–Osher ;
- temps d'exécution ;
- `teacher_fidelity` : NN ≤ 1.02× TENO5 ;
- `beats_teacher` : NN ≤ TENO5 contre la même vérité terrain.

Seul `beats_teacher` autorise le mot **gain**.

---

## 12. Validation et tests

## 12.1 Ce que vérifient les tests

- reconstruction exacte d'un état constant ;
- advection périodique finie et respect CFL ;
- validation des conditions aux limites ;
- convergence WENO3/WENO5/WENO7 en région lisse ;
- stabilité des masques TENO et fallback tout-rejeté ;
- normalisation des stencils constants ;
- formes et ordre du feature vector ;
- matrices propres de Roe mutuellement inverses ;
- régressions Sod et Shu–Osher ;
- égalité NumPy/JAX en float64 ;
- finitude de toutes les pertes et de leurs gradients ;
- cohérence projection soft/hard et mélange convexe.

## 12.2 État vérifié lors de la création de ce document

Commande numérique exécutée :

```bash
python -m pytest tests/test_core.py tests/test_teno.py tests/test_feature_vector.py -q
```

Résultat :

```text
33 passed, 33 subtests passed
```

Le shell courant utilise Python 3.13 sans Keras ni JAX. La collecte de toute la suite échoue donc sur les imports de ces deux stacks ; ce n'est pas un échec numérique. Les tests spécialisés doivent être lancés dans leurs environnements respectifs.

## 12.3 Commandes de reproduction

### Numérique sans réseau

```bash
pip install -r requirements-numerical.txt
python -m pytest tests/test_core.py tests/test_teno.py tests/test_feature_vector.py -q
```

### Keras historique

```bash
conda create -n weno-nn python=3.7
conda activate weno-nn
pip install -r requirements.txt
python scripts/legacy_imitation_training/Train_NNTENO5.py --device cpu
```

### JAX

```bash
conda create -n teno-jax python=3.11
conda activate teno-jax
pip install -r requirements-jax.txt
python -m pytest \
  tests/test_teno5_jax_regression.py \
  tests/test_teno5_jax_euler_regression.py \
  tests/test_projection_jax.py \
  tests/test_losses_jax.py -q
```

### Entraînement physique et comparaisons

```bash
python scripts/unrolled_physics_loss/Train_NNTENO5_UnrolledJAX.py
python scripts/unrolled_physics_loss/Train_NNTENO5euler_UnrolledJAX_Curriculum.py
python scripts/unrolled_physics_loss/Train_NNTENO5_UnrolledJAX_LearnedLambda.py
python scripts/unrolled_physics_loss/Train_NNTENO5euler_UnrolledJAX_LearnedLambda.py
python scripts/unrolled_physics_loss/ValidationMatrix_ProjPDF.py
python scripts/unrolled_physics_loss/TENO5euler_vs_NNTENO5euler_JAX.py
```

---

## 13. Ce qui a réellement fonctionné

1. **Mesurer contre la vraie solution.** Sans cela, le réseau ne peut pas recevoir de signal pour dépasser le professeur.
2. **Préentraîner par imitation.** L'imitation n'est pas la solution finale, mais c'est une excellente initialisation stable.
3. **Ancrer les poids.** Cela empêche un fine-tuning agressif de détruire instantanément les propriétés apprises.
4. **Normaliser les pertes.** Une somme dépendant de la taille du maillage ou du réseau ne doit pas être comparée naïvement à une moyenne.
5. **Utiliser `lax.scan`.** Nécessaire pour rendre la compilation du rollout réaliste.
6. **Pénaliser avant réparation.** Une pénalité évaluée après avoir caché l'erreur est mathématiquement inactive.
7. **Réparer localement les violations isolées.** La moyenne des voisins est plus physique qu'un clamp aveugle.
8. **Borner les magnitudes et clipper les gradients.** La finitude des valeurs de loss ne garantit pas celle des gradients.
9. **Conserver TENO5 dans l'architecture.** Le mélange convexe transforme TENO5 en point d'ancrage fonctionnel, pas seulement en professeur historique.
10. **Séparer sélection et confiance.** Une tête dédiée \(\lambda\) évite de demander au même signal d'accomplir deux tâches contradictoires.
11. **Valider au vrai horizon canonique.** Un gain à \(K=15\) ne garantit pas un gain à \(T=1\) ou \(T=1.8\).
12. **Sauvegarder le meilleur checkpoint.** Un dernier batch anormal peut détruire une longue trajectoire d'entraînement correcte.

---

## 14. Ce qui n'a pas fonctionné et pourquoi

| Tentative | Pourquoi elle paraissait logique | Pourquoi elle a échoué |
|---|---|---|
| imiter WENO/TENO plus longtemps | labels faciles et stables | aucune information sur la vraie erreur |
| cibles exactes locales 5 points | supprime le professeur | la cible n'est pas toujours identifiable localement |
| passer simplement à 7 points | plus de contexte | contexte utile mais objectif/compromis de stabilité toujours insuffisant |
| oracle one-shot | choix local optimal | erreurs locales réinjectées dans la trajectoire, blow-up |
| augmenter seulement la dissipation LF | remède classique à l'instabilité | réponse non monotone ; cause plus structurelle |
| entraînement physique à froid | objectif final direct | signal long-horizon mal conditionné et réseau initial instable |
| clamp de positivité final | garde un état valide | cache les violations et arrive trop tard pour les sous-étapes |
| floor \(10^{-8}\) | constante positive standard | crée vitesses et sons gigantesques pour des états d'échelle O(1) |
| pénalité sur état réparé | terme présent dans la loss | toujours exactement nulle |
| curriculum seul | facilite l'optimisation | retarde la divergence sans supprimer sa cause |
| ajout de données lisses sans clipping | élargit la distribution | un gradient extrême détruit le checkpoint |
| entraîner directement avec \(\lambda=\max p\) | mélange différentiable gratuit | sélection et confiance partagent les mêmes paramètres |
| rendre seulement la porte plus complexe | davantage de capacité de calibration | le backbone reste inférieur sur Sod |

---

## 15. Limites honnêtes

- Gains modestes : le plus petit est environ 1.8 % sur Shu–Osher.
- Sod n'est pas gagné ; le meilleur compromis reste environ 0.35 % moins précis.
- La référence Shu–Osher est numérique, pas analytique.
- Résultats limités au 1D et au cinquième ordre.
- Distribution d'entraînement finie ; la robustesse très hors distribution reste inconnue.
- Réseau plus lent que le schéma classique dans l'implémentation actuelle : environ 5–18× selon le pipeline historique, sans kernel fusionné.
- Le mécanisme de réparation de positivité contient un fallback de finitude qui n'est pas présenté comme une solution physique exacte pour un cluster instable.
- La porte enrichie n'a pas résolu Sod ; de nouvelles variantes de porte sont peu prometteuses sans changer le backbone.
- Les stacks Keras et JAX sont séparées et les checkpoints ne sont pas interchangeables automatiquement.
- Les chemins absolus historiques stockés dans certains JSON pointent vers l'ancien emplacement du dépôt ; les scripts actuels résolvent normalement les I/O depuis la racine.

---

## 16. Perspectives justifiées par les résultats

### Priorité 1 — améliorer le backbone sur Sod

- sur-échantillonner des problèmes de Riemann proches de Sod ;
- inclure explicitement raréfaction, contact et choc dans l'objectif de validation ;
- essayer un backbone plus large ou résiduel ;
- fournir plus de contexte spatial ou temporel ;
- conserver la tête Phase 18 comme garde-fou pendant ce travail.

### Priorité 2 — garanties structurelles

- projection différentiable imposant simultanément positivité, conservation et éventuellement entropie ;
- contrôle de la condition de la base propre de Roe ;
- fallback local adaptatif vers TENO5 avant que l'état ne quitte le domaine physique.

### Priorité 3 — généralisation

- familles d'initialisation plus larges ;
- Woodward–Colella, blast waves et Sedov ;
- tests sur des horizons beaucoup plus longs ;
- seeds multiples et intervalles de confiance.

### Priorité 4 — coût

- batching/fusion de toutes les interfaces ;
- compilation JIT de la chaîne complète ;
- comparaison coût–erreur à budget égal ;
- distillation de la tête et du backbone ;
- profilage GPU et CPU.

### Priorité 5 — extension scientifique

- TENO7 différentiable après correction complète de ses singularités ;
- reconstruction multi-D ;
- problèmes visqueux ;
- maillages non uniformes ;
- tests sur données CFD industrielles.

---

## 17. Trame de présentation orale

### Version 12–15 minutes

1. **Contexte et problème — 1 min**

   Les chocs exigent un compromis précision/stabilité ; WENO/TENO utilisent une règle analytique locale.

2. **Base numérique — 2 min**

   Volumes finis, SSPRK3, flux splitting, WENO vs TENO, reconstruction caractéristique Euler.

3. **Baseline imitation — 2 min**

   WENO5-NN puis NN-TENO5. Bonne fidélité mais plafond du professeur.

4. **Contribution principale — 3 min**

   Port JAX, rollout différentiable, quatre termes principaux de loss plus la positivité Euler, projection soft/hard.

5. **Difficulté Euler — 2 min**

   Positivité, sous-étapes, base de Roe, pénalité inactive, réparation, curriculum, clipping.

6. **Mélange convexe — 2 min**

   Pourquoi garder TENO5 dans la boucle ; tête dédiée \(\lambda\).

7. **Résultats et limites — 2 min**

   Gains scalaires et Shu–Osher, quasi-égalité sur Sod, coût, 1D, généralisation.

8. **Conclusion — 30 s**

   Le signal physique et les contraintes structurelles comptent plus qu'une simple augmentation de capacité.

### Trois messages à répéter

1. « Je compare toujours les deux méthodes à la même vérité terrain. »
2. « Les résultats négatifs ont servi de diagnostics et ont directement déterminé l'expérience suivante. »
3. « Le réseau n'est utile que lorsqu'il reste encadré par la structure du schéma classique. »

---

## 18. Questions probables du jury et réponses

### Pourquoi un réseau si TENO5 fonctionne déjà ?

TENO5 utilise une frontière de décision analytique fixe. Un réseau peut exploiter les statistiques d'une famille de problèmes pour corriger localement un compromis trop dissipatif ou trop agressif. Le projet montre que ce potentiel existe, mais seulement avec une cible physique et une architecture contrainte.

### Pourquoi ne pas prédire directement le flux ?

Une prédiction libre facilite les erreurs non conservatrices ou instables. NN-TENO ne modifie que la sélection des candidats déjà conçus analytiquement. Le mélange convexe va plus loin : il garde TENO5 comme référence sûre.

### Pourquoi l'imitation ne peut-elle pas dépasser le professeur ?

Sa fonction objectif récompense uniquement la proximité avec le professeur. Un éventuel gain ne peut être qu'accidentel. Pour chercher systématiquement une erreur physique plus faible, la vérité terrain doit apparaître dans la loss.

### Le réseau d'imitation gagne pourtant parfois. Contradiction ?

Non. Une approximation imparfaite du professeur peut lisser avantageusement certaines décisions et produire un gain accidentel. Mais la loss ne pousse pas de manière fiable dans cette direction, d'où les gains petits et non systématiques.

### Pourquoi utiliser JAX ?

Le pipeline NumPy/Keras appelle `model.predict`, ce qui coupe le graphe de gradient. JAX rend différentiables la reconstruction, le flux, SSPRK3 et le rollout complet, et `lax.scan` évite de compiler K copies séparées.

### Pourquoi soft pendant l'entraînement et hard à l'inférence ?

La décision binaire a un gradient nul presque partout. La version soft fournit un chemin de gradient continu. La version hard restitue ensuite la logique TENO de garder/rejeter.

### Pourquoi SSPRK3 ?

Il est d'ordre 3, courant dans les solveurs hyperboliques et construit à partir de combinaisons convexes de pas Euler. Il offre un bon compromis stabilité/coût et correspond au pipeline historique.

### Pourquoi la base de Roe est-elle fragile ?

Elle dépend de \(\rho\), \(p\), vitesse et enthalpie. Si une reconstruction produit un état proche du vide ou extrême, les divisions, racines et inversions deviennent mal conditionnées. L'erreur affecte alors la base du pas suivant et crée une boucle de rétroaction.

### Pourquoi ne pas seulement augmenter la dissipation ?

L'expérience a montré une réponse non monotone : 1.5× retardait l'explosion, mais 2–5× la faisaient arriver plus tôt. Cela réfute l'hypothèse d'une simple borne de vitesse trop petite.

### La réparation de positivité ne fausse-t-elle pas la physique ?

La moyenne des voisins n'est utilisée que pour une violation isolée. Un groupe de violations est traité comme une instabilité et pénalisé. Le fallback clampé sert uniquement à garder le graphe numérique fini, pas à déclarer la trajectoire physiquement correcte.

### Pourquoi le terme de positivité était-il nul ?

Il était calculé après la réparation. Il ne voyait donc jamais la violation. La correction accumule la pénalité sur chaque sous-étape brute avant réparation.

### Pourquoi normaliser TV et Reg ?

Leur magnitude brute augmente respectivement avec le nombre de cellules et le nombre de paramètres, contrairement à la MSE moyenne. Sans normalisation, les coefficients lambda ne sont pas comparables et TV dominait réellement le gradient.

### Pourquoi le mélange convexe aide-t-il ?

Il limite l'amplitude de la déviation par rapport à un schéma stable. Le réseau fournit une correction faible là où elle est utile, sans prendre entièrement le contrôle de la reconstruction.

### Pourquoi une tête dédiée ?

La probabilité « troubled » répond à « faut-il rejeter ce candidat ? ». \(\lambda\) répond à « combien faire confiance à la reconstruction globale du réseau ? ». Utiliser le même signal pour les deux crée une interférence. Une tête séparée découple les objectifs.

### Pourquoi Sod reste-t-il moins bon ?

Le sweep de \(\lambda\) montre que tout mélange non nul du backbone augmente monotoniquement l'erreur Sod. Le problème n'est donc pas que la porte choisit mal \(\lambda\), mais que la proposition du backbone est elle-même moins bonne que TENO5 sur ce cas.

### Le gain Shu–Osher est-il significatif ?

Il est petit, 1.84 %, donc il ne faut pas le sur-vendre. Il est toutefois mesuré au même maillage, au même horizon et contre la même référence, et il apparaît avec un checkpoint stable et physiquement valide. C'est un gain réel mais étroit.

### Le modèle est-il plus rapide ?

Non. Dans les implémentations actuelles, les variantes neuronales coûtent environ 5–18× plus cher. Le projet démontre une possibilité d'amélioration d'erreur, pas encore un avantage coût–performance.

### Pourquoi seulement 1D ?

Le but du stage est d'isoler le mécanisme scientifique. En 1D, les références exactes ou convergées permettent un diagnostic propre. La 2D ajouterait trop de facteurs avant de comprendre la stabilité de base.

### Comment garantir la reproductibilité ?

Environnements séparés et figés, seeds, métadonnées JSON, checkpoints, scripts d'évaluation indépendants, références exactes lorsqu'elles existent, résultats négatifs conservés et tests de régression NumPy/JAX en float64.

---

## 19. Glossaire express

| Terme | Définition |
|---|---|
| CFL | rapport entre distance parcourue par une onde pendant un pas et taille de cellule |
| cellule moyenne | valeur intégrale moyenne stockée par volumes finis |
| stencil | voisinage de cellules utilisé pour reconstruire une interface |
| reconstruction | estimation d'une valeur/flux à l'interface |
| ENO | choisit un sous-stencil lisse |
| WENO | combine continûment plusieurs candidats |
| TENO | garde/rejette les candidats par séparation d'échelle |
| indicateur de régularité | mesure des variations/curvatures d'un candidat |
| flux splitting | séparation du flux selon le sens de propagation |
| Roe | linéarisation locale caractéristique d'Euler |
| BVD | variation entre deux reconstructions voisines à une interface |
| TVD | propriété empêchant la création de variation totale |
| rollout | déroulement du solveur pendant plusieurs pas |
| warm start | initialisation depuis un checkpoint appris |
| ancrage L2-SP | pénalité maintenant les poids proches du checkpoint initial |
| backbone | réseau principal produisant les décisions de reconstruction |
| mixing head | petit réseau apprenant le coefficient \(\lambda\) |
| OOD | hors distribution d'entraînement |
| checkpoint | état sauvegardé des paramètres |
| exact-mask accuracy | part des exemples où les trois décisions sont toutes correctes |

---

## 20. Fichiers à ouvrir pendant la soutenance

- Vue générale : [`../README.md`](../README.md)
- Journal exact des 21 phases : [`Results.md`](Results.md)
- Guide de la perte déroulée : [`UNROLLED_PHYSICS_LOSS_GUIDE.md`](UNROLLED_PHYSICS_LOSS_GUIDE.md)
- Correspondance avec la littérature : [`Literature_Report_Mapping.md`](Literature_Report_Mapping.md)
- Rapport français : [`Rapport_de_Stage_FR.tex`](Rapport_de_Stage_FR.tex)
- TENO5 NumPy : [`../src/schemes/teno5.py`](../src/schemes/teno5.py)
- Solveur JAX scalaire : [`../src/differentiable/numerics_jax.py`](../src/differentiable/numerics_jax.py)
- Solveur JAX Euler : [`../src/differentiable/euler_jax.py`](../src/differentiable/euler_jax.py)
- Loss complète : [`../src/differentiable/losses_jax.py`](../src/differentiable/losses_jax.py)
- Mélange/projection : [`../src/differentiable/projection_jax.py`](../src/differentiable/projection_jax.py)
- Backbone et têtes : [`../src/differentiable/teno5_jax_toy.py`](../src/differentiable/teno5_jax_toy.py)
- Comparaison finale scalaire : [`../scripts/unrolled_physics_loss/ValidationMatrix_ProjPDF.py`](../scripts/unrolled_physics_loss/ValidationMatrix_ProjPDF.py)
- Comparaison finale Euler : [`../scripts/unrolled_physics_loss/TENO5euler_vs_NNTENO5euler_JAX.py`](../scripts/unrolled_physics_loss/TENO5euler_vs_NNTENO5euler_JAX.py)

---

## 21. Conclusion prête à prononcer

> Ce stage a montré qu'ajouter un réseau à un schéma de capture de choc ne suffit pas. Un réseau entraîné par imitation apprend surtout à reproduire la décision du schéma classique. Pour obtenir un vrai gain, il faut l'entraîner contre la solution physique à travers la trajectoire du solveur. Cette approche a demandé un port JAX différentiable, une gestion correcte de la positivité et des gradients, puis une combinaison convexe qui conserve TENO5 comme référence sûre. Le résultat final est un gain mesuré sur l'advection, Burgers et Shu–Osher, avec une limite clairement identifiée sur Sod. La contribution principale n'est donc pas l'affirmation qu'un réseau bat toujours TENO5, mais l'identification expérimentale des conditions sous lesquelles il peut le faire, des mécanismes qui rendent l'entraînement stable, et des limites qui exigent encore un meilleur backbone.

---

## 22. Références principales

1. G.-S. Jiang et C.-W. Shu, *Efficient Implementation of Weighted ENO Schemes*, Journal of Computational Physics, 1996.
2. L. Fu, X. Y. Hu et N. A. Adams, *A family of high-order targeted ENO schemes for compressible-fluid simulations*, Journal of Computational Physics, 2016.
3. Z. Sun, F. Inaba et F. Xiao, *Boundary Variation Diminishing reconstruction*, Journal of Computational Physics, 2016.
4. B. Stevens et T. Colonius, *Enhancement of shock-capturing methods via machine learning*, Theoretical and Computational Fluid Dynamics, 2020.
5. G. Li, L. Fu et N. A. Adams, travaux NENO6 sur la sélection neuronale de stencils, 2022.
6. K. Um et al., *Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers*, NeurIPS, 2020.
7. Documents fournis avec le dépôt : [`../references/WENO-NN.pdf`](../references/WENO-NN.pdf), [`../references/TENO_ML.pdf`](../references/TENO_ML.pdf), [`../references/Unrolled_Physics_Loss_Proposal.pdf`](../references/Unrolled_Physics_Loss_Proposal.pdf).
