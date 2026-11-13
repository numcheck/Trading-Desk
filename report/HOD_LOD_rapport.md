# Estimer et trader le HOD / LOD de la séance RTH — ES & NQ

Document de travail. Code : `hodlod/`. Auto-tests : `python -m hodlod.selftest` (21/21 passent).

> **Mise à jour — résultats empiriques (voir `research/out/HOD_LOD_ES_NQ.pdf`).**
> Le protocole des sections 8 à 10 a été exécuté sur données réelles gratuites
> (Yahoo Finance) : 1 383 séances ES & NQ en quotidien (2021-2026), 491 séances
> × 3 instruments en horaire (2024-2026). Résultats mesurés, pas simulés :
>
> | Ce qui est testé | Résultat | Verdict |
> |---|---|---|
> | Amplitude `E[range]/σ` (théorie 1,596) | **1,80 – 1,82** | la diffusion sous-estime de ~13 % |
> | Multiplicateur d'amplitude q25→q95 | **1,06 – 1,12**, plat | correction par une constante `κ ≈ 1,09` |
> | Bande `[q25,q75]` de `R+` (nominal 50 %) | **50,3 %** | calibré |
> | Arcsinus en horloge brute | **rejeté** (χ² 34,5 / 53,5 / 108,8) | le nul est faux |
> | Arcsinus en temps de vol | χ² réduit de 40 %, KS ÷ 2,6 | mieux, mais **toujours rejeté** |
> | `P(HOD courant = final)` vs climatologie horaire | **BSS +0,33 à +0,39** | la formule prédit réellement |
> | Backtest walk-forward, 1 190 trades | brut **+0,0236 R**, coût **0,0313 R**, net **−0,0076 R** | edge réel, **sous la friction** |
> | DSR (90 essais enregistrés) | **0,002** | rien de prouvé |
>
> Conséquence pour la section 11 : l'estimation « 25–40 % de chances d'un edge
> réel » était la bonne fourchette, et le point dur n'est pas le signal mais le
> **coût par unité de risque** : il faudrait descendre sous 1 % de σ par
> aller-retour pour que le même signal devienne rentable.

---

## 0. Hypothèses explicites (à défaut de questions)

| # | Hypothèse |
|---|---|
| H1 | Univers : ES et NQ, front-month continu ajusté par différence, barres 1 min, RTH 09:30–16:00 ET. `O` = premier trade ≥ 09:30, `H`/`L` = max/min du high/low sur [09:30, 16:00], `C` = dernier trade ≤ 16:00. |
| H2 | ≥ 2 500 séances (≈ 10 ans) de 1-min. En dessous de ~1 000 séances, tout modèle à plus de 3–4 paramètres est indécidable, quelle que soit la p-value affichée. |
| H3 | `σ_day` = écart-type de la séance RTH complète. Il n'est connu qu'ex post : tout le problème consiste à le **prévoir**, puis à le **mettre à jour**, et à travailler en unités de σ. |
| H4 | Aucune donnée de marché n'est disponible dans cet environnement. Les **fréquences empiriques** (taux de rejet des niveaux, PIT réel, edge) ne sont donc **pas** fournies : je donne l'estimateur, son intervalle de confiance et le contrôle placebo. Inventer ces chiffres serait la faute la plus grave possible ici. |
| H5 | Coûts : ES 0,25 pt = 12,50 $/tick, 50 $/pt ; NQ 0,25 pt = 5 $/tick, 20 $/pt. Aller-retour au marché ≈ 1,5–2 ticks tout compris. Range RTH typique ES 40–60 pts (160–240 ticks) → coût ≈ 0,8–1,2 % du range ; NQ 250–400 pts (1 000–1 600 ticks) → ≈ 0,15 %. **En ticks relatifs, NQ est plus tradable ; ES a un meilleur rapport signal/bruit par point.** |
| H6 | La dérive intraday n'est pas supposée nulle. Toutes les formules sont données avec et sans. |
| H7 | `HOD_final` n'est connu qu'à 16:00. Le label « le max courant est le final » est donc un label de fin de journée : utilisable seulement avec un décalage d'un jour dans l'entraînement. |
| H8 | Session unique, pas de Globex dans les features (sauf le range overnight, connu à 09:29). |

---

## 1. Le problème, énoncé correctement

Le HOD n'est pas un niveau, c'est un **argmax**. Trois conséquences qui commandent toute la conception :

1. **Il n'y a pas de « niveau du HOD » à prédire.** Il y a une distribution de `(H−O)/σ` et une distribution de l'instant `T_H`. Chercher un prix exact est une erreur de spécification.
2. **Deux questions distinctes.** Ex ante : « où sera le max final ? » → une distribution. En temps réel : « le max courant est-il le final ? » → une probabilité de survie. Seule la seconde est actionnable pendant la séance.
3. **L'événement est absorbant.** `1{HOD_t = HOD_final}` vaut 1 dès que le max est fait, donc sa probabilité monte mécaniquement vers 1 en fin de séance. C'est un problème de **survie**, pas de classification. Toute métrique de type AUC sera flatteuse et trompeuse (cf. §8).

Le seul contenu informationnel réellement exploitable est donc :
- l'**amplitude** (combien la séance peut parcourir),
- l'**asymétrie** (quel côté s'étend),
- le **timing** (quand le max est probablement fait).

L'amplitude est très prévisible (c'est de la volatilité). L'asymétrie l'est très peu. C'est là que se joue l'edge, et c'est là qu'il est mince.

### Identité de calibration la plus utile

Pour un MB sans dérive sur [0,T], `E[max] = σ√T·√(2/π) = 0.7979·σ√T` et `E[range] = √(8/π)·σ√T = 1.5958·σ√T`. Donc :

```
σ_day  ≈  range_RTH / 1.596
```

Un ES qui range 50 points a `σ_day ≈ 31 points`. Cette identité sert de contrôle de cohérence permanent entre vos trois sources de σ (range, RV 1-min, HAR).

---

## 2. Classement par pertinence réelle pour localiser HOD/LOD

Note de 1 à 5. « Rôle réel » = ce que le modèle fait vraiment, pas ce qu'on aimerait qu'il fasse.

| # | Modèle | Ce qu'il estime exactement | Note | Rôle réel |
|---|---|---|---|---|
| 1 | **Survie de maximum / premier passage** (§6) | `P(sup_{[t,T]} X < M_t | F_t)` | **5** | Le seul objet qui répond à la question temps réel. C'est le déclencheur. |
| 2 | **CARR** (Chou 2005) | `λ_t = E[H_t−L_t | F_{t−1}]`, distribution conditionnelle complète via `ε` | **4** | Échelle conditionnelle de l'amplitude. Dénominateur de tout le reste. |
| 3 | **HAR-RV** (Corsi 2009) | `E[RV_{t+1} | RV_t, RV̄_5, RV̄_22]` | **4** | Intrant indispensable (`σ_day`). Ne localise **rien**. |
| 4 | **Régression quantile / QRF / boosting pinball** | `q_α((H−O)/σ | X)`, `q_α((O−L)/σ | X)` | **4** | La carte pré-ouverture. Le seul modèle qui donne une région. |
| 5 | **Loi de l'arcsinus en temps de volatilité** | `P(T_H ≤ t) = (2/π)arcsin√F(t)` | **3,5** | Prior de timing gratuit, zéro paramètre. Benchmark à battre. |
| 6 | **Barrières à deux côtés** (range) | `P(−a < min, max < b)` | **3** | Probabilité que le reste de séance tienne dans un couloir → fade de range. |
| 7 | **EVT / GPD sur amplitude normalisée** | loi de la queue de `R±` au-delà d'un seuil | **3** | Placement de stop, taille de queue, pas la localisation. |
| 8 | **Niveaux de confluence + placebo** (§5) | fréquence de rejet conditionnelle à la distance | **3** | Filtre du déclencheur. **Non validé sans contrôle placebo.** |
| 9 | **Estimateurs de range** (Parkinson, GK, RS, YZ) | `σ²` du jour, à partir de OHLC | **2,5** | Instruments de **mesure**. Améliorent le label RV, pas la prévision. |
| 10 | **HEAVY / Realized GARCH** | variance conditionnelle + mesure réalisée | **2,5** | Alternative à HAR pour `σ_day`. Même rôle. |
| 11 | **CAViaR** (Engle & Manganelli 2004) | `q_θ(r_t | F_{t−1})` — un quantile de **rendement** | **1,5** | **Autre chose : la VaR.** Utile pour dimensionner le stop, inutile pour localiser H/L. |

### Ce qui sert à autre chose (à dire clairement)

- **CAViaR** → VaR / Expected Shortfall. Le quantile d'un rendement n'est pas le max d'un chemin. Confondre les deux est l'erreur la plus fréquente dans ce type de projet.
- **EVT** → queues de rendements, VaR extrême, stress-testing. Réutilisable pour l'amplitude **uniquement** après normalisation par σ (sinon on mesure la saisonnalité de la vol, pas la queue).
- **Yang-Zhang** → estimation de variance pour le pricing d'options et la prévision de vol. C'est le meilleur des quatre en efficacité, et le seul qui survive aux gaps, mais il exige une fenêtre (n ≥ 2) et ne dit rien sur H/L.
- **HAR-RV / HEAVY** → prévision de volatilité. Excellent à ce qu'il fait (R² OOS 0,4–0,6 en log), sans contenu directionnel.
- **Parkinson / GK / RS** → mesure. Leur intérêt ici est indirect : un meilleur `RV_t` améliore le HAR, qui améliore `σ_day`, qui améliore toutes les probabilités.

> **Point dur :** aucun de ces modèles ne « localise » le HOD au sens « voici le prix ». Ils produisent une région et une probabilité. Un système qui prétend donner un prix exact n'a pas de base statistique.

---

## 3. A — Amplitude : `(H−O)/σ` et `(O−L)/σ`

### A.0 Le benchmark exact, et il est gratuit

Sous MB sans dérive, `H−O` a la loi de `|N(0, σ²T)|` (conséquence directe du principe de réflexion) :

```
P(H−O ≤ x) = 2Φ(x / S) − 1 ,     S = σ√T
```

D'où, sans aucun paramètre à estimer :

| Quantité | Valeur |
|---|---|
| `E[(H−O)/S]` | `√(2/π) = 0.7979` |
| médiane | `Φ⁻¹(0.75) = 0.6745` |
| `q_90` | `Φ⁻¹(0.95) = 1.6449` |
| `q_95` | `Φ⁻¹(0.975) = 1.9600` |
| `E[(H−O)+(O−L)]/S` | `1.5958` (= `√(8/π)`, l'espérance du range) |

Tout écart systématique à cette loi est une **prime de queues / sauts / dérive** qu'il faut mesurer, pas supposer. Le diagnostic le plus informatif est le **multiplicateur d'amplitude** :

```python
# hodlod/amplitude.py
amplitude_multiplier(R_plus_standardise)  # -> {q10: m10, ..., q95: m95}
```

Lecture :
- ratio **plat > 1** → simple erreur d'échelle sur σ (votre σ est trop petit) ;
- ratio **croissant avec α** → queues épaisses (sauts, vol stochastique) ;
- ratio **> 1 aux deux bouts** → bornes de monitoring + sauts.

**Résultat mesuré (Monte-Carlo, 8 000 chemins, 1 200 points de monitoring)** : la CDF empirique du max vaut 0,6933 contre 0,6827 en théorie à `x = 1σ`. Autrement dit, **le monitoring discret sous-estime l'amplitude d'environ 1 %** (on ne voit pas les extrêmes entre deux barres). Corollaire pratique : la formule continue est légèrement **optimiste** ; calibrez un facteur de monitoring `κ` tel que `S_eff = κ·σ√T` avec `κ ≈ 0,99` en 1-minute, `κ → 1` en tick.

### A.1–A.4 Estimateurs de range : formules, hypothèses, où ça casse

| Estimateur | Formule | Efficacité vs C2C | Hypothèses | Où ça casse |
|---|---|---|---|---|
| Parkinson (1980) | `σ²_P = (ln(H/L))² / (4 ln2)` | 5,2× | MB sans dérive, **sans gap**, monitoring continu | gap d'ouverture → biais vers le bas ; dérive → biais vers le haut |
| Garman-Klass (1980) | `σ²_GK = ½(ln H/L)² − (2ln2−1)(ln C/O)²` | 7,4× | idem + pas de gap | idem |
| Rogers-Satchell (1991) | `σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)` | ~8× | MB **avec dérive** autorisée, sans gap | gap d'ouverture |
| Yang-Zhang (2000) | `σ²_YZ = σ²_on + k·σ²_oc + (1−k)·σ²_RS`, `k = 0.34/(1.34+(n+1)/(n−1))` | ~14× | dérive + gaps, fenêtre n ≥ 2 | peu de choses ; le plus robuste, mais c'est une moyenne de fenêtre, pas un estimateur du jour |

**Message central : ce sont des instruments de mesure, pas des prévisions.** Ils répondent à « quelle était la vol du jour ? », pas à « quel sera le range de demain ». Leur usage correct ici :
1. construire un `RV_t` propre (barres 1-min) et un `RV_t` robuste aux sauts (`BV = (π/2)Σ|r_t||r_{t−1}|`, Barndorff-Nielsen & Shephard 2006) ;
2. fournir des régresseurs au HAR (`log RP_prev`, `log GK_prev`, `log RS_prev`) ;
3. contrôler la cohérence `σ_day ≈ range/1.596`.

### A.5 EVT / GPD sur l'amplitude normalisée

**Ce qu'il estime :** la loi de la queue de `R⁺ = (H−O)/σ_day`, au-delà d'un seuil `u`.

```
u = q_F(R⁺) ;  Y = R⁺ − u | R⁺ > u  ~  GPD(ξ, β)
P(R⁺ > u+y) = (1−F_u)·(1 + ξy/β)^(−1/ξ)
q_α(R⁺) = u + (β/ξ)[ ((1−α)/(1−F_u))^(−ξ) − 1 ]
```

**Données :** 2 500+ séances de `R⁺` **normalisées par σ_day prévu** (jamais par le σ réalisé : ce serait du look-ahead).
**Calibration :** seuil au 90ᵉ centile ; vérifier la **stabilité de ξ** sur une grille de seuils (graphique de stabilité) ; `scipy.stats.genpareto.fit(exc, floc=0)`.
**Hypothèses :** queue à variation régulière, exces approximativement iid.
**Où ça casse :** (i) le choix du seuil pilote ξ ; (ii) les exces sont **clusterisés** (jours de crise) → il faut déclusteriser ou utiliser un modèle conditionnel ; (iii) la non-stationnarité : si vous ne normalisez pas, vous mesurez la saisonnalité de la vol, pas la queue ; (iv) à 2 500 obs il reste ~250 exces, l'IC sur ξ est d'environ `±(1+ξ)/√250 ≈ ±0,08` — large.

**Validation :** couverture de `q_99` et de `q_995` (doit être 99 % et 99,5 % ± IC binomial), et perte pinball au-delà du seuil.
**Mesuré sur simulation Pareto(ξ=1/3), n=50 000 :** ξ estimé 0,374 (vrai 0,333), `q_99` à +0,7 % — l'estimateur est correct mais ξ est bruité.

### A.6 HAR-RV

```
RV_{t+1} = c + β_d·RV_t + β_w·RV̄_{t−4..t} + β_m·RV̄_{t−21..t} + ε
```

En pratique on estime sur `log RV` (stabilise l'hétéroscédasticité) et on retransforme avec la correction de smearing `exp(σ̂²_ε/2)`.

**Données :** `RV_t = Σ r²_{1min}` (ou `BV` pour la partie continue). Version `HARQ` (Bollerslev, Patton & Quaedvlieg 2016) : ajouter `RQ_t^{1/2}` pour corriger le biais d'erreur de mesure — gain réel sur les données hautes fréquences.
**Calibration :** OLS ; erreurs HAC. Pour l'inférence sur `log RV`, l'OLS suffit.
**Hypothèses :** structure hiérarchique (commerçants à horizons 1j/1sem/1mois), coefficients stables, RV sans bruit de microstructure.
**Où ça casse :** ruptures de régime (mars 2020), roll de contrat (un RV calculé sur un prix non ajusté explose le jour du roll), demi-séances, et barres 1-min sur NQ où le bruit de microstructure est plus fort.
**Validation :** QLIKE et MSE sur RV hors échantillon, plus une régression Mincer-Zarnowitz `RV = a + b·RV̂` (testez `b=1`).

### A.7 CARR — et la réponse à votre question sur l'article fondateur

**Vous avez raison.** L'article fondateur est :

> Chou, R. Y. (2005). *Forecasting Financial Volatilities with Extreme Values: The Conditional Autoregressive Range (CARR) Model.* **Journal of Money, Credit and Banking, 37(3), 561–582.** DOI 10.1353/mcb.2005.0027.

Ce que **Chen, Gerlach & Lin (2008)** est réellement :

> Chen, C. W. S., Gerlach, R., & Lin, E. M. H. (2008). *Volatility forecasting using threshold heteroskedastic models of the intra-day range.* **Computational Statistics & Data Analysis, 52(6), 2990–3010.**

C'est une **extension à seuil** de CARR (asymétrie signe/taille, estimation bayésienne MCMC), pas le papier fondateur. Attribuer CARR à Chen et al. (2008) est une erreur de citation.

**Modèle (Chou 2005) :**

```
R_t = λ_t · ε_t ,        ε_t iid, E[ε] = 1, ε ⊥ F_{t−1}
λ_t = ω + Σ_{i=1..p} α_i R_{t−i} + Σ_{j=1..q} β_j λ_{t−j}
```

Extension `CARR-X` (même article) : `+ Σ γ_k x_{k,t−1}` pour des exogènes (VIX, RV overnight).

**Ce qu'il estime exactement :** `λ_t = E[R_t | F_{t−1}]`, l'amplitude **attendue du range**, plus la loi conditionnelle complète via `ε`. Ce n'est **pas** une prévision de variance, même si `λ_t` en est un proxy monotone.

**Distribution conditionnelle :** si `ε ~ Exp(1)`, alors `q_α(R_t) = −λ_t ln(1−α)`. C'est immédiatement utilisable comme carte d'amplitude.

**Calibration :** quasi-ML (la vraisemblance exponentielle reste convergente si `ε` n'est pas exponentiel, tant que `E[ε]=1`).

```python
from hodlod.vol import CARR
m = CARR().fit(R)                      # R = range quotidien en points
lam_next = m.predict_next(R)           # E[range de demain | histoire]
q80 = CARR.quantile(lam_next, 0.80)    # q80 du range de demain
m.persistence, m.se_                   # alpha+beta, erreurs-types (BHHH)
```

**Hypothèses :** linéarité en `R`, pas de seuil, `ε` de moyenne 1 et iid.
**Où ça casse :** (i) jours d'événement — le range explose sans que `λ_t` l'ait anticipé ; (ii) `β → 1` ⇒ sous-réaction aux ruptures ; (iii) l'asymétrie documentée par Chen et al. (2008) : un choc baissier n'a pas le même effet qu'un choc haussier sur le range ; (iv) le range est borné par la liquidité de la session (demi-séances).
**Validation :** QLIKE sur le range (`QLIKE = R/λ − ln(R/λ) − 1`), et perte pinball des quantiles `λ_t·(−ln(1−α))`.

**Vérification numérique :** sur 20 000 observations simulées avec `(ω,α,β) = (0.25, 0.10, 0.85)`, le QMLE retrouve `(0.0999, 0.8583)`, persistance 0,958 contre 0,95. L'implémentation est correcte.

### A.8 Le modèle d'amplitude qu'on oublie : la partie **restante**

Le range déjà parcouru consomme le budget. À 11:00, l'espérance de `H−O` conditionnelle à un range déjà grand est **plus faible**, pas plus forte. Le bon objet est donc :

```
R⁺_restant = (H_final − H_t) / σ_remaining ,    σ_remaining = σ_day·√(1 − F(t))
```

qui est **exactement** la même fonction de survie que celle du §6. Il n'y a pas deux modèles : amplitude et temps réel sont le même objet mathématique vu de deux côtés. C'est le point d'architecture le plus important du document.

---

## 4. B — Localisation

### B.1 Probabilité « quel côté d'abord » (avec dérive)

Sous MB avec dérive totale `M` sur l'horizon et écart-type total `S`, partant de l'open :

```
P(toucher +b avant −a) = (1 − e^(−2Ma/S²)) / (1 − e^(−2M(a+b)/S²))
```

À `M = 0` : `a/(a+b)`. C'est la seule formule du document qui contient une **direction**, et elle est entièrement déterminée par la dérive. Toute l'asymétrie que vous pourrez trader vient de là, ou n'existe pas.

### B.2 Régression quantile (Koenker & Bassett 1978)

```
q_α(Y | X) = argmin_θ  Σ ρ_α(Y_i − X_i'θ),   ρ_α(u) = u(α − 1{u<0})
```

**Cibles :** `R⁺ = (H−O)/σ̂` et `R⁻ = (O−L)/σ̂`. Quantiles utiles : 0,25 / 0,50 / 0,75 / 0,90.
**Features (toutes connues avant 09:30) :** `log RV_1,5,22`, `log RP_prev`, `range_prev/σ`, `gap/σ`, `range_overnight/σ`, `VIX`, `ΔVIX`, jour de semaine, `is_event` (FOMC/CPI/NFP, calendrier connu ex ante), position de la clôture de la veille dans son range.
**Calibration :** `statsmodels.QuantReg`. Écarts-types : bootstrap par blocs (jours), pas les SE asymptotiques.
**Hypothèses :** linéarité, quantiles non croisés, stationnarité.
**Où ça casse :** croisements de quantiles (fréquents à 0,90), hétéroscédasticité non capturée, et surtout **peu d'observations efficaces** : 2 500 jours, 12 features, 4 quantiles → la variance d'estimation domine vite.
**Validation :** perte pinball hors échantillon, **couverture** de l'intervalle `[q05, q95]` (doit être 90 %), et `CRPS = 2∫₀¹ QL_τ dτ`.
**Mesuré sur simulation :** couverture 0,906 pour un `q_0.9` nominal à 0,90.

### B.3 Quantile Regression Forests et boosting pinball

- **QRF** (Meinshausen 2006) : forêts aléatoires + estimation des quantiles conditionnels par pondération des observations. Consistant, non paramétrique, gère les interactions. Risque : lent, et les quantiles extrêmes sont mal estimés (les feuilles contiennent peu d'obs).
- **Boosting pinball** : `sklearn.GradientBoostingRegressor(loss="quantile", alpha=α)`. Plus rapide, meilleur sur les interactions. Régler `max_depth ≤ 3`, `min_samples_leaf ≥ 40`, `subsample = 0.8`.
- Un modèle **par quantile** indépendant ⇒ croisements. Vérifier avec `amplitude.crossing_check` et, si le taux de croisement dépasse 5 %, réarranger les prédictions (tri par quantile).

**Le vrai gain du ML ici n'est pas la précision moyenne : c'est la dépendance au régime** (jour d'événement × vol élevée × gap). Une régression linéaire ne capte pas `is_event × vol`.

### B.4 Classifieur d'asymétrie — le seul signal directionnel du cadre

Cible : `1{R⁺ > R⁻}`, c'est-à-dire « la branche haute domine-t-elle ? ».

Sous MB sans dérive : **exactement 0,5**. Il n'y a donc rien à trouver, sauf dérive, autocorrélation ou microstructure.

Le point de calibrage des attentes : Gao, Han, Li & Zhou (2018) documentent un **intraday momentum** robuste (le rendement de la première demi-heure prédit celui de la dernière) avec un **R² hors échantillon de 1,4 % à 2,0 %**, et jusqu'à 3,3 % les jours de forte vol. C'est un effet publié dans le *Journal of Financial Economics*, répliqué sur 16 marchés — et son R² est de 2 %. C'est l'ordre de grandeur de **tout** ce que vous pouvez espérer sur l'asymétrie.

Corollaire : un modèle d'asymétrie avec un R² de 15 % est un bug, pas une découverte.

### B.5 Niveaux de confluence : la seule méthode qui ne se mente pas à elle-même

Niveaux : `PDH`, `PDL`, `PDC`, `PD VWAP`, `ONH`, `ONL`, `IBH`, `IBL` (connus à 10:00), `VWAP` courante, `POC`/`VAH`/`VAL`, chiffres ronds, high/low hebdo et mensuel.

**Le piège :** « le PDH est touché 70 % du temps » ne veut rien dire. Un niveau proche du prix est touché plus souvent, mécaniquement. Trois obligatoires :

1. **Conditionner sur la distance** `|level − O|/σ_day`.
2. **Contrôle placebo à distance appariée** : tirer un niveau factice à la même distance (distribution empirique des distances réelles) et calculer la même statistique. Si l'écart n'est pas significatif, le niveau n'apporte rien.
3. **Définitions symétriques** : touch = mèche dans `[L−tol, L+tol]` ; cassure = clôture au-delà de `L+tol` ; rejet = touch sans cassure **et** retour à plus de `3·tol` du côté d'origine.

```python
from hodlod.levels import confluence_table
confluence_table(days, "pdh", tol_sigma=0.05, placebo=True)
# -> {'n':..., 'reject_rate':..., 'placebo_reject_rate':..., 'delta':..., 'delta_ci95':[...], 'significant':bool}
```

**Le seul chiffre interprétable est `delta` et son IC.** Le reste est du bruit déguisé en statistique.

**Ce qui casse :** les niveaux sont corrélés entre eux (PDH et ONH coïncident souvent), donc un test par niveau surestime la significativité globale ; il faut un test joint (Bonferroni ou SPA de Hansen 2005), et un `n` effectif qui tient compte du clustering par régime.

---

## 5. C — Timing

### C.1 La loi de référence, gratuite et testable

Troisième loi de l'arcsinus (Lévy 1939/1948) : pour un MB sans dérive sur [0,1],

```
P(T_H ≤ t) = (2/π)·arcsin(√t)          densité 1/(π√(t(1−t)))
```

Chiffres à retenir :

| `t` | `P(T_H ≤ t)` |
|---|---|
| 0,10 | **0,205** |
| 0,25 | 0,333 |
| 0,50 | 0,500 |
| 0,75 | 0,667 |
| 0,90 | 0,795 |

**41 % des séances placent leur high dans les 10 % extrêmes de la journée** (première ou dernière 48 minutes). Ce n'est pas une régularité empirique de la vol intraday : c'est une propriété de **tout** marché aléatoire continu. Elle est donc un prior, pas une anomalie.

Idem pour `T_L`, et `P(T_H < T_L) = 1/2` sans dérive.

### C.2 Validation empirique sur ES

> Mok, D. M. Y., Lam, K., & Li, W. (2000). *Using Daily High/Low Time to Test for Intraday Random Walk in Two Index Futures Markets.* **Review of Quantitative Finance and Accounting, 14, 381–397.**

Ils dérivent la loi théorique de l'instant du high/low sous random walk (à vitesse de trading uniforme **ou** variable), et concluent que **pour le S&P 500 futures, l'hypothèse de random walk n'est pas rejetée**, alors qu'elle l'est pour le Hang Seng (extrêmes trop souvent dans les 15 premières minutes). C'est la référence la plus directement pertinente pour votre univers, et elle est plutôt défavorable : sur ES, le timing du high/low est déjà bien décrit par le modèle nul.

### C.3 Temps de volatilité : `F(t)` plutôt que `t`

Le profil saisonnier intraday (Andersen & Bollerslev 1998) n'est pas une correction cosmétique, c'est un **changement d'horloge**. Définissez

```
s(t) = profil de variance normalisé (Σ s = 1),   F(t) = ∫₀ᵗ s(u)du
```

Alors le prior de timing devient `P(T_H ≤ t) = (2/π)arcsin√F(t)`, et `σ_remaining(t) = σ_day·√(1−F(t))`.

**Mesuré (Monte-Carlo, profil en U marqué, 4 000 chemins, 1 000 points)** : en horloge brute, la distance KS à la loi de référence est **0,101** ; en temps de volatilité, **0,031**. Le gain est d'un facteur 3,3 — réel mais **pas parfait** : le changement de temps est une approximation, et le profil est estimé, pas connu. Ne présentez jamais cette transformation comme exacte.

**Piège de diagnostic :** sous H0, `F(T_H)` suit une loi **arcsinus**, pas une loi uniforme. Le PIT correct est `u = ArcsinCDF(F(t_max))`. Oublier la seconde transformation fait rejeter H0 à tort — c'est l'erreur la plus fréquente dans ce diagnostic (et je l'ai commise dans la première version du code, les tests l'ont attrapée).

### C.4 Hasard et risques concurrents

Version non paramétrique, directement comparable au modèle de diffusion :

```
h_k = P(premier dépassement dans le bin k | pas encore dépassé)
S_k = Π_{j≤k}(1−h_j)
```

`S_k` est la probabilité que le max courant soit encore le final au bin `k`. C'est la cible d'un classifieur et le contrôle du modèle analytique.

`timing.competing_risk_table` donne en plus `P(H et L déjà tous deux fixés)`, qui définit le **régime de fade de range** : une stratégie structurellement différente du retournement d'extrême, et probablement plus robuste.

### C.5 Où le timing casse

1. **Annonces à heure fixe** : 10:00 (ISM, confiance), 14:00 (FOMC), 14:30 (conférence de presse), 8:30 (CPI/NFP, avant l'open). Le hasard n'est pas lisse : il a des pics déterministes.
2. **Enchères** : l'ouverture (09:30) et la clôture (15:50–16:00) ont une microstructure propre. Le « high de la dernière minute » est souvent un artefact d'enchère.
3. **Demi-séances** (13:00 ET) : `F(t)` change de forme. Les exclure ou les traiter séparément.
4. **Intraday momentum** : les jours de tendance font un nouveau plus haut dans les 30 dernières minutes plus souvent que l'arcsinus ne le prédit. C'est précisément l'effet de Gao et al. (2018).

---

## 6. D — Temps réel : « le HOD courant est-il le final ? »

### D.1 La formule, sa dérivation, et ses cinq façons d'échouer

Soit `M_t` le max courant, `X_t` le prix, `d = M_t − X_t ≥ 0`, `S = σ·√τ` l'écart-type du prix sur l'horizon restant. Pour un MB sans dérive partant de `X_t` :

```
P(sup_{[t,T]} X < M_t | F_t) = P(sup_{u≤τ} (X_t + σW_u) < X_t + d)
                            = P(sup W < d/σ√τ)
                            = 2Φ(d/S) − 1
```

C'est exactement la formule de votre énoncé. Elle est **exacte** sous ses hypothèses, et elle a deux limites utiles à connaître : `d→0 ⇒ p→0`, `d/S→∞ ⇒ p→1` (lentement, en `1 − 2φ(z)/z`).

**Avec dérive** (formule de la barrière absorbante) :

```
P(max < a) = Φ((a−M)/S) − e^(2Ma/S²)·Φ((−a−M)/S)          M = dérive totale sur l'horizon
```

Vérifié : `M→0` redonne `2Φ(a/S)−1` (erreur mesurée 3e-13).

**Les cinq violations, et leur direction :**

| # | Violation | Effet sur `p` | Ordre de grandeur |
|---|---|---|---|
| 1 | **Dérive** | dépend du signe ; une dérive haussière **baisse** `p_H` | `M/S ≈ 0,03` pour la dérive actions moyenne (négligeable) ; `0,5–1` les jours de tendance (décisif) |
| 2 | **Queues épaisses / sauts** | `p` **surestimé** (un saut peut franchir une barrière lointaine) | le terme dominant les jours de news |
| 3 | **Vol stochastique** | même direction (mélanger des σ épaissit les queues) → `p` surestimé | `p_final_mixture(vol_of_vol=0,35)` |
| 4 | **Monitoring discret** | `p` **sous-estimé** (peu d'occasions de franchir) | **+0,015 mesuré** en 1 000 points |
| 5 | **Autocorrélation / momentum** | `ρ>0` ⇒ `p` sous-estimé | l'effet Gao et al. |

Les effets 2/3 et 4 vont en sens **opposé** et se compensent partiellement. C'est pour cela que la formule « marche mieux que prévu » en pratique — et c'est une coïncidence, pas une validation. Ne l'utilisez jamais sans recalibration.

**Vérification Monte-Carlo (5 000 chemins, 2 000 points)** :

| Test | Formule | Empirique |
|---|---|---|
| sans dérive | 0,4906 | 0,5060 |
| dérive +0,4 | 0,3628 | 0,3690 |
| dérive −0,4 | 0,6221 | 0,6360 |

L'écart systématique (+0,010 à +0,015) est le monitoring discret, exactement comme prévu. La structure conditionnelle est correcte.

### D.2 La couche qui rend l'objet utilisable

```
p_final(t) = g( p_diffusion(d_t, S_t),  features_t )
```

où `g` est une **calibration isotonique** estimée en walk-forward, et où les features sont :
`d/S`, `F(t)`, `σ̂` (niveau), `vol_ratio = RV_réalisée / RV_attendue`, `range_so_far/σ`, `asym_amp`, `dist_vwap/σ`, `log volume`, pente de tendance, distance aux niveaux de confluence, proximité d'une annonce, type de jour.

**Le piège de l'AUC.** La base rate de `1{HOD_t = HOD_final}` monte vers 1 en fin de séance. Un classifieur qui apprend « il est 15:55 » obtient une AUC de 0,97 sans aucune information. Il faut donc :
- évaluer le Brier **à instants fixés** (10:00, 11:00, 12:30, 14:00, 15:00),
- comparer au **modèle analytique**, pas à la climatologie,
- rapporter l'AUC mais ne jamais l'optimiser.

Formule obligatoire :

```
BSS = 1 − BS(modèle) / BS(diffusion)
```

Un `BSS > 0` signifie « mieux que la théorie ». C'est le seul résultat qui vaut quelque chose.

### D.3 Version à deux côtés (fade de range)

`p_range_within(a, b, S)` donne la probabilité que le reste de séance tienne dans `[X−a, X+b]`. Série d'images à deux barrières :

```
P(−a < min, max < b) = Σ_k [Φ((b+2kL)/S) − Φ((−a+2kL)/S)] − Σ_k [Φ((b+2a+2kL)/S) − Φ((a+2kL)/S)]
```

avec `L = a+b`. **La seconde somme est indispensable** : l'oublier donne 0,68 au lieu de 0,37 pour `a=b=S` (bug trouvé par les tests). Validé en Monte-Carlo à 0,371 contre 0,380 (l'écart est le monitoring discret).

---

## 7. Références — vérifiées une par une

### Articles à comité de lecture (vérifiés)

| Référence | Revue / vol / pages | Vérif. |
|---|---|---|
| Chou, R. Y. (2005). Forecasting Financial Volatilities with Extreme Values: The Conditional Autoregressive Range (CARR) Model | *J. Money, Credit and Banking* 37(3), 561–582 | ✅ **fondateur CARR** |
| Chen, C. W. S., Gerlach, R., Lin, E. M. H. (2008). Volatility forecasting using threshold heteroskedastic models of the intra-day range | *Computational Statistics & Data Analysis* 52(6), 2990–3010 | ✅ extension à seuil, **pas** le fondateur |
| Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return | *Journal of Business* 53(1), 61–65 | ✅ |
| Garman, M. B., Klass, M. J. (1980). On the Estimation of Security Price Volatilities from Historical Data | *Journal of Business* 53(1), 67–78 | ✅ |
| Rogers, L. C. G., Satchell, S. E. (1991). Estimating Variance from High, Low and Closing Prices | *Annals of Applied Probability* 1(4), 504–512 | ✅ |
| Yang, D., Zhang, Q. (2000). Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices | *Journal of Business* 73(3), 477–492 | ✅ |
| Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility | *J. Financial Econometrics* 7(2), 174–196 | ✅ |
| Bollerslev, T., Patton, A. J., Quaedvlieg, R. (2016). Exploiting the errors: A simple approach for improved volatility forecasting (HARQ) | *Journal of Econometrics* 192(1), 1–18 | ✅ |
| Engle, R. F., Manganelli, S. (2004). CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles | *J. Business & Economic Statistics* 22(4), 367–381 | ✅ (WP NBER 7341, 1999) |
| Koenker, R., Bassett, G. (1978). Regression Quantiles | *Econometrica* 46(1), 33–50 | ✅ |
| Meinshausen, N. (2006). Quantile Regression Forests | *JMLR* 7, 983–999 | ✅ |
| Shephard, N., Sheppard, K. (2010). Realising the future: forecasting with high-frequency-based volatility (HEAVY) models | *J. Applied Econometrics* 25(2), 197–231 | ✅ |
| Andersen, T. G., Bollerslev, T. (1998). Deutsche Mark–Dollar Volatility: Intraday Activity Patterns, Macroeconomic Announcements, and Longer Run Dependencies | *Journal of Finance* 53(1), 219–265 | ✅ |
| Mok, D. M. Y., Lam, K., Li, W. (2000). Using Daily High/Low Time to Test for Intraday Random Walk in Two Index Futures Markets | *Review of Quantitative Finance and Accounting* 14, 381–397 | ✅ **directement sur S&P futures** |
| Gao, L., Han, Y., Li, S. Z., Zhou, G. (2018). Market intraday momentum | *J. Financial Economics* 129(2), 394–414 | ✅ |
| Holmberg, U., Lönnbark, C., Lundström, C. (2013). Assessing the profitability of intraday opening range breakout strategies | *Finance Research Letters* 10(1), 27–33 | ✅ |
| Bailey, D. H., López de Prado, M. (2014). The Deflated Sharpe Ratio | *J. Portfolio Management* 40(5), 94–107 | ✅ |
| Bailey, D. H., Borwein, J. M., López de Prado, M., Zhu, Q. J. (2017). The Probability of Backtest Overfitting | *J. Computational Finance* 20(4), 39–69 | ✅ |
| Gneiting, T., Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation | *JASA* 102(477), 359–378 | ✅ |
| Brier, G. W. (1950). Verification of forecasts expressed in terms of probability | *Monthly Weather Review* 78(1), 1–3 | ✅ |

### Théorie (manuels)

- Lévy, P. (1939, 1948) — lois de l'arcsinus. Voir Karatzas & Shreve (1991), *Brownian Motion and Stochastic Calculus*, problème 8.16 : `P(θ_t ≤ s) = (2/π)arcsin√(s/t)`.
- Feller, W. (1971), *An Introduction to Probability Theory and Its Applications*, vol. II — barrières, loi du range.
- Borodin, A. N., Salminen, P. (2002), *Handbook of Brownian Motion: Facts and Formulae* — séries d'images à deux barrières.
- López de Prado, M. (2018), *Advances in Financial Machine Learning*, ch. 7 — purging et embargo.

### Travaux récents **non** à comité de lecture (à traiter comme tels)

Ces trois-là comptent pour l'estimation honnête de l'edge, pas comme preuve :

- Fetna, M. (2026). *Opening-Range Breakout Does Not Survive Trading Costs: A Pre-Registered 225-Cell Study on Sixteen Years of Futures Data.* SSRN 7428398. → 0 cellule sur 225 ne passe le seuil ; +1,57 M$ brut, −6,49 M$ net à 25 $ l'aller-retour ; **edge brut médian −0,01 tick**.
- Chuk, J. (2026). *Regime-Dependent Opening Range Breakout Returns in Nasdaq-100 Futures: A Pre-Registered Study.* Working paper. → 1 survivant sur 144 tests pré-enregistrés sur NQ.
- Subhani, S. (2026). *Opening-Range Breakouts on Event-Free Fridays.* SSRN 7589980. → DSR 0,78 (QQQ) / 0,42 (SPY) après déflation sur 57 configurations.

**Ce que ces papiers disent vraiment :** sur votre univers exact (ES/NQ, 1-min, 2010–2026), la famille « breakout de range d'ouverture » produit un edge brut **inférieur au coût de transaction**. C'est la base rate à garder en tête pour toute stratégie d'extrême intraday.

---

## 8. Protocole de validation

### 8.1 Découpage

```
purged_walk_forward(n_jours, train=1500, test=125, embargo=5, label_horizon=0)
```

Le label `HOD_final` est connu à 16:00 ⇒ pour prédire un événement du jour J, on s'entraîne sur ≤ J−1. Aucune exception.

### 8.2 Checklist anti-look-ahead (spécifique à ce problème)

- [ ] `σ_day` est une **prévision**, jamais `RV_J`. Tester : corréler `σ̂_J` avec `RV_J` ; si la corrélation dépasse ~0,6 hors échantillon, il y a une fuite.
- [ ] Le profil `F(t)` est estimé sur une fenêtre glissante **ne contenant pas J** (ou sur une moyenne historique figée).
- [ ] `VWAP_t` n'utilise que les barres `≤ t`.
- [ ] `IBH/IBL` sont `NaN` avant 10:00.
- [ ] `is_event` vient d'un calendrier **connu ex ante** (les dates FOMC/CPI sont publiées).
- [ ] Le roll de contrat est décidé sur volume/OI **avec un jour de retard**.
- [ ] Fuseau horaire : 09:30 ET n'est pas une heure UTC fixe. Convertir en `America/New_York`, pas en UTC−4/−5.
- [ ] Demi-séances exclues ou traitées séparément.
- [ ] Les jours où le high est fait dans la dernière minute sont exclus de l'évaluation de `1{HOD final}` (label dégénéré).
- [ ] `leakage_audit(feature_names)` ne renvoie rien.

### 8.3 Métriques, par sous-problème

| Sous-problème | Métrique principale | Métrique secondaire | Seuil d'acceptation |
|---|---|---|---|
| A. Amplitude | perte pinball par quantile, CRPS | couverture `[q05,q95]` | `skill > 0` vs CARR ; couverture 90 % ± 2 |
| A. σ_day | QLIKE sur RV | Mincer-Zarnowitz `b=1` | `R²_OOS(log RV) > 0,35` |
| B. Asymétrie | log-loss | Brier skill vs 0,5 | `BSS > 0` |
| C. Timing | log-score discret | KS du PIT | KS < 0,05 (n ≈ 2 500) |
| D. Temps réel | **Brier skill vs diffusion** | ECE, pente de calibration | `BSS > 0,05` |
| Économie | P&L net, Sharpe | max DD, turnover | voir §8.4 |

### 8.4 Économie et sur-ajustement

- **Coûts** : 1,5 / 2,0 / 3,0 ticks aller-retour, en scénarios. Un résultat qui n'existe qu'à 1 tick n'existe pas.
- **Bootstrap par blocs** (bloc = 5 jours) pour l'IC du P&L moyen : préserve l'autocorrélation.
- **Test SPA** (Hansen 2005) ou reality check (White 2000) si vous comparez plusieurs variantes.
- **DSR** (Bailey & López de Prado 2014) avec `n_trials` **réellement enregistré**. Sans registre des essais, le DSR est incalculable et le résultat est indéfendable.
- **PBO** (Bailey et al. 2017) par CSCV si vous avez moins de 20 variantes.
- **Règle de décision** : un Sharpe de 1,0 sur 500 trades a un IC à 95 % d'environ `[0,1 ; 1,9]`. Il ne prouve rien. Le nombre de trades efficaces, pas le nombre de trades, est ce qui compte : avec des trades corrélés intra-jour, divisez par 3 à 5.

---

## 9. Architecture complète

```
                    ┌─────────────────────────────────────────┐
  J-1 16:05         │ POST-SESSION                            │
                    │ • RV, BV, range, P/GK/RS                 │
                    │ • label HOD/LOD, T_H, T_L, asymétrie     │
                    │ • mise à jour du profil F(t) (glissant)  │
                    │ • mise à jour HAR/CARR + calibration     │
                    └──────────────────┬──────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
  J-1 17:00→09:15   │ PRÉ-OUVERTURE                           │
                    │ • σ̂_day = HAR(log RV, HARQ) × exp(γ'X)  │
                    │ • σ̂_residual = σ̂_day·√(1−F(t))          │
                    │ • carte d'amplitude q10..q90 (R⁺, R⁻)   │
                    │ • niveaux : PDH/PDL/PDC/ONH/ONL/VWAP    │
                    │ • P(trend up/down/range) ; is_event     │
                    └──────────────────┬──────────────────────┘
                                       ▼
  J 09:25           │ PLAN FIGÉ                               │
                    │ zone H attendue = O + [q50, q85]·σ̂      │
                    │ zone L attendue = O − [q50, q85]·σ̂      │
                    │ bande de non-trade = |q25| de part et d'autre
                                       ▼
  J 09:30           │ OUVERTURE                               │
                    │ • gap, position vs ONH/ONL, vs PD VWAP  │
                    │ • si |gap| > 0.5σ̂ ⇒ re-pondérer le plan │
                                       ▼
  J 10:00           │ INITIAL BALANCE                         │
                    │ • IBH/IBL disponibles                   │
                    │ • mise à jour bayésienne de σ̂ :          │
                    │   σ̂² ← (1−w)·σ̂² + w·RV_30min/F(0.5h)    │
                    │   w ≈ 0.3 (poids du signal 30 min)      │
                                       ▼
  Chaque minute      │ ÉTAT TEMPS RÉEL                         │
                    │ • d_H = M_t − P_t ; d_L = P_t − m_t     │
                    │ • p_H = g(2Φ(d_H/S_t)−1, features)      │
                    │ • p_L = g(2Φ(d_L/S_t)−1, features)      │
                    │ • vol_ratio, dist_vwap, niveaux         │
                                       ▼
        ┌──────────────────────────────────────────────┐
        │ DÉCLENCHEUR DE RETOURNEMENT (short au H)     │
        │  p_H ≥ 0,80                                   │
        │  ET P_t ≥ M_t − 0,15·σ̂                        │
        │  ET confluence (niveau réel OU VWAP) à 0,1σ̂   │
        │  ET signature de rejet : nouveau high fait    │
        │     puis clôture < M_t sur 3 barres           │
        │  ET 10:00 ≤ t ≤ 15:30                         │
        │  ET vol_ratio ∈ [0,7 ; 2,0]  (pas de trend)   │
        └───────────────────┬──────────────────────────┘
                            ▼
        ┌──────────────────────────────────────────────┐
        │ RISQUE                                        │
        │ stop  = M_t + max(1,5·ATR5m, 0,25·σ̂)          │
        │ cible = VWAP puis q50 de la branche opposée   │
        │ taille = budget_risque / distance_stop        │
        │ 1 tentative par niveau (pas de re-entrée)     │
        │ max 2 trades/jour, perte max 2R/jour          │
        │ sortie forcée 15:50                           │
        └───────────────────┬──────────────────────────┘
                            ▼
                    ┌─────────────────────────────────────────┐
                    │ JOURNAL + MONITORING                    │
                    │ • Brier / calibration par tranche        │
                    │ • P&L par niveau, par heure, par régime  │
                    │ • kill switch si BSS < 0 sur 20 séances  │
                    └─────────────────────────────────────────┘
```

**Trois décisions d'architecture qui comptent plus que le modèle :**

1. **Un seul objet mathématique** pour l'amplitude restante et la probabilité de survie (§A.8). Deux modèles séparés divergent.
2. **La mise à jour bayésienne de σ̂ à 10:00** est le gain le plus fiable de tout le système : la vol des 30 premières minutes contient plus d'information sur le reste de séance que tout le modèle pré-ouverture.
3. **Le kill switch est une partie du système, pas une option.** `BSS < 0` sur 20 séances consécutives ⇒ arrêt, pas « ajustement des paramètres ».

---

## 10. Plan de test concret

| Étape | Test | Critère d'échec → action |
|---|---|---|
| **T0** | Intégrité données : stitching du continu, fuseau `America/New_York`, demi-séances, minutes manquantes, doublons | > 0,5 % de séances avec un timestamp suspect ⇒ corriger avant tout |
| **T1** | Audit du label : recalculer H/L par une seconde agrégation ; compter la part de HOD dans les 5 dernières minutes | écart > 1 tick ⇒ bug d'agrégation |
| **T2** | **Réplication du nul** : loi de l'arcsinus sur `F(T_H)` (KS) ; demi-normalité de `R⁺` (multiplicateur) | KS < 0,05 attendu ; si KS > 0,15, le profil `F(t)` est mal estimé |
| **T3** | Amplitude : walk-forward, pinball vs CARR et vs demi-normal | skill ≤ 0 ⇒ le modèle n'apporte rien, revenir à CARR |
| **T4** | Temps réel : Brier à 10:00/11:00/12:30/14:00/15:00, **vs formule de diffusion** | BSS ≤ 0 ⇒ pas d'edge sur la probabilité, arrêter ici |
| **T5** | Niveaux : `delta` avec placebo apparié, correction Bonferroni sur ~8 niveaux | `significant=False` ⇒ retirer le filtre de confluence |
| **T6** | Économie : coûts 1,5/2/3 ticks, bootstrap blocs, DSR avec registre des essais | DSR < 0,95 ⇒ ne pas trader |
| **T7** | Stabilité : signe du P&L identique sur ES et NQ, sur 2 sous-périodes, et monotone dans le seuil `p_H` | non-monotone ⇒ artefact de seuil, pas un edge |
| **T8** | Paper trading 3 mois, taille minimale | écart de plus de 50 % au backtest ⇒ le backtest est faux |

**Le test T2 est le plus important et le moins coûteux.** Si vous ne reproduvez pas la loi de l'arcsinus et la demi-normalité sur vos données, votre pipeline a un bug, et aucune quantité de ML ne le corrigera.

---

## 11. Estimation honnête de l'edge

### Ce qui est solide

- **Prévoir la volatilité** : oui, franchement. `R²_OOS(log RV) ≈ 0,4–0,6`. C'est réel, robuste, répliqué. Mais c'est de la **taille**, pas de la **direction**.
- **Le prior de timing** : réel, gratuit, et… public et sans paramètre. Il n'apporte pas d'edge, il apporte une **meilleure calibration**, ce qui est déjà beaucoup pour ne pas trader au mauvais moment.
- **La probabilité de survie** : l'objet est correct et la formule exacte sous ses hypothèses. La couche de calibration au-dessus est le seul endroit où un edge peut apparaître.

### Ce qui est mince

- **L'asymétrie** : le R² de l'effet publié le plus solide (Gao et al. 2018) est de **1,4–2,0 % hors échantillon**. Attendez-vous à un déplacement de quelques points de probabilité, pas à un signal net.
- **Le retournement au HOD** a une **asymétrie négative** : vous vendez un nouveau high dans une journée qui peut être un trend day. Les pertes sont plus grandes que les gains, la fréquence de réussite doit être élevée pour compenser.
- **La base rate de la famille** : les études pré-enregistrées 2026 sur l'ORB en ES/NQ trouvent un edge brut médian de **−0,01 tick**. C'est le point de départ réaliste.

### Chiffrage

Hypothèses : ES, `σ_day = 31 pts`, stop à `0,25σ ≈ 8 pts` (R = 8 pts = 400 $), cible 1R, coût 2 ticks = 25 $ = **6 % de R**.

| Version | Trades/sem. | Espérance brute/trade | Net | Sharpe annualisé attendu |
|---|---|---|---|---|
| Naïve (fade de chaque touche de la bande σ) | 8–12 | ≈ 0 | **négatif** (−6 % de R) | < 0, certain |
| Filtrée (p_H ≥ 0,8 + confluence + rejet) | 2–4 | 0,05–0,15 R | 0 à +0,10 R | 0,2–0,6 |
| Filtrée + régime (vol, événement, tendance) | 1–2 | 0,10–0,25 R | +0,05–0,20 R | 0,3–0,8 |

**Ce que ces chiffres valent vraiment :** avec 300 trades, l'IC à 95 % du Sharpe est environ `[−0,1 ; 0,9]`. Statistiquement, on ne peut pas distinguer 0,4 de 0. Il faut **500 à 1 000 trades** (3 à 5 ans à 2–4/semaine) pour trancher.

**Probabilité qu'une implémentation soignée ait un edge réellement positif : 25–40 %.** Ce n'est pas du pessimisme, c'est la base rate de la famille, coûts inclus.

**Dégradation attendue backtest → live : 40–60 %** du Sharpe. Causes : le fill des limites dans la mèche (vous n'êtes pas seul à vendre le nouveau high), le slippage sur les journées de tendance, et le fait que le backtest suppose que vous êtes là à chaque signal.

### Ce qui me ferait croire au résultat

1. `BSS > 0,05` contre la formule de diffusion, hors échantillon, sur 500+ séances.
2. Le résultat survit à **3 ticks** aller-retour.
3. **DSR > 0,95** avec `n_trials` enregistré.
4. Le signe est le même sur **ES et NQ**, et sur **deux sous-périodes**.
5. L'effet est **monotone** dans le seuil `p_H` (0,6 → 0,7 → 0,8 → 0,9), pas un décrochage à un seuil magique.

Si l'un de ces cinq manque, ce n'est pas un edge, c'est un backtest.

---

## 12. Code

| Fichier | Contenu |
|---|---|
| `hodlod/vol.py` | Parkinson, GK, RS, Yang-Zhang ; profil saisonnier ; temps de volatilité ; HAR-RV (log, smearing) ; CARR(1,1) quasi-ML avec erreurs-types BHHH ; RV et BV |
| `hodlod/amplitude.py` | benchmark demi-normal + multiplicateur ; GPD/POT (fit, quantile, tail prob) ; régression quantile ; `QuantileBooster` ; détection de croisements ; classifieur d'asymétrie |
| `hodlod/timing.py` | lois de l'arcsinus (cdf/pdf/ppf) ; temps de volatilité ; PIT à deux étapes ; KS ; log-score ; hasard empirique ; risques concurrents |
| `hodlod/realtime.py` | `p_no_new_high` (sans dérive), `p_no_new_high_drift`, `p_range_within` (deux barrières), moyenne sur vol stochastique, `remaining_sigma`, `session_features`, calibration isotonique |
| `hodlod/levels.py` | niveaux de référence, initial balance, touch/break/reject, **placebo à distance appariée**, `confluence_table` avec IC bootstrap, volume profile approché |
| `hodlod/evaluate.py` | `purged_walk_forward`, Brier/BSS/log-loss/ECE/AUC, pinball/CRPS/couverture, bootstrap par blocs, PSR/DSR, `leakage_audit` |
| `hodlod/selftest.py` | 21 vérifications Monte-Carlo de toutes les formules fermées |

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m hodlod.selftest      # 21/21
```

### Ce que les tests ont attrapé (et pourquoi c'est la partie la plus utile)

| Bug | Conséquence si non détecté |
|---|---|
| Série à deux barrières sans les images réfléchies | `P(range)` surestimée de 84 % (0,68 au lieu de 0,37) |
| PIT testé contre l'uniforme au lieu de l'arcsinus | rejet systématique d'un modèle correct |
| Simulation CARR sans terme `α` | validation d'un modèle sans mémoire, et faux sentiment que CARR ne marche pas |
| Indexation booléenne scalaire (NumPy 2) | crash silencieux de la branche avec dérive |
| Indentation cassée par un patch | `ImportError` à l'import |

C'est l'argument pour écrire les tests avant le backtest : sur cinq bugs, **quatre étaient dans ma propre validation**, pas dans les modèles.

---

## 13. Ce que je ferais dans les 10 premiers jours

1. **Jour 1–2** : T0 + T1. Nettoyer les données. Rien d'autre.
2. **Jour 3** : T2. Répliquer l'arcsinus et la demi-normalité. Si ça échoue, tout le reste est faux.
3. **Jour 4–5** : HAR + CARR, `σ̂_day`, QLIKE, `R²_OOS`. C'est le socle.
4. **Jour 6–7** : carte d'amplitude par régression quantile, pinball, couverture.
5. **Jour 8** : `p_final` de diffusion brute + calibration isotonique. **BSS contre la formule.**
6. **Jour 9** : T5, niveaux avec placebo. C'est le filtre le plus susceptible de surestimer son propre pouvoir.
7. **Jour 10** : décision. Si le BSS de l'étape 5 est ≤ 0, **arrêter** — il n'y a pas de signal dans la probabilité, et aucun filtre de niveau ne le créera.
