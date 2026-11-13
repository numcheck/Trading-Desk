# Trading-Desk

Recherche quantitative intraday sur **ES** (S&P 500) et **NQ** (Nasdaq 100).

---

## Où se forment le HOD et le LOD — étude statistique et backtest

> **Verdict en une phrase :** le modèle statistique est bon, l'edge de trading est
> réel mais **plus petit que les frais**. Espérance brute **+0,024 R** par trade,
> coût aller-retour **0,031 R**, net **−0,008 R** sur 1 190 trades hors échantillon.

📄 **[Lire le rapport complet (PDF, 11 pages)](research/out/HOD_LOD_ES_NQ.pdf)**

Le high-of-day (HOD) et le low-of-day (LOD) ne sont pas des niveaux mais des
**argmax**. Ce dépôt traite les quatre sous-problèmes séparément — amplitude,
localisation, timing, probabilité temps réel — puis les agrège en une stratégie
et la teste honnêtement.

![Amplitude contre le benchmark de diffusion](research/out/fig/f_amplitude.png)

### Les chiffres à retenir

| Question | Théorie | Mesuré | Lecture |
|---|---|---|---|
| Amplitude `E[range]/σ` | 1,596 | **1,820** | le marché parcourt 14 % de plus que la diffusion |
| Multiplicateur d'amplitude q25→q95 | 1,00 | **1,06–1,12** | une seule constante `κ ≈ 1,09` corrige tout |
| `P(amplitude dans [q25,q75])` | 50 % | **50,3 %** | calibré |
| Loi de l'arcsinus (horloge) | — | **rejetée** (χ² 34–109) | le marché se concentre plus à l'ouverture |
| Loi de l'arcsinus (temps de vol) | — | KS ÷ 2,6, **toujours rejetée** | mieux, mais insuffisant |
| `P(HOD courant = final)` | — | **BSS +0,33 à +0,39** | la formule prédit réellement |
| Backtest walk-forward | — | **−0,0076 R net** | brut positif, sous la friction |

![Timing du high](research/out/fig/f_timing.png)

La question « combien de fois le marché fait-il ce qu'on calcule » a sa propre
section dans le PDF. Extrait, sur les 370 séances exploitables (σ̂ disponible) :

| Le HOD est-il déjà final à… | Prédit | Observé |
|---|---|---|
| 10:30 | 34,7 % | **35,1 %** |
| 12:30 | 49,1 % | **56,5 %** |
| 15:30 | 76,9 % | **81,8 %** |

L'écart est systématiquement négatif : le modèle est **légèrement sous-confiant**,
exactement la direction prédite par le biais de monitoring discret.

![Courbe hors échantillon](research/out/fig/f_equity.png)

### Ce que le test ne dit pas

- **Granularité.** La source gratuite plafonne l'intraday à **730 jours** (5/15/30 min :
  60 jours seulement). Le backtest est donc horaire. Un déclencheur à la minute
  aurait un timing d'entrée très différent : c'est la limite principale.
- **Instruments.** Le backtest tourne sur SPY/QQQ/IWM (proxies dont le OHLC est
  validé à 2 bp près), pas sur les futures. Les coûts appliqués sont ceux d'ES.
- **Aucun résultat n'est ajusté en faveur de la stratégie.** Le registre des
  90 configurations testées est dans `research/out/results.json`.

### Structure

```
hodlod/            Librairie théorique (aucune dépendance aux données)
├── vol.py           Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang,
│                    profil saisonnier, HAR-RV, CARR(1,1) quasi-ML
├── amplitude.py     benchmark demi-normal, GPD/POT, régression quantile, boosting
├── timing.py        lois de l'arcsinus, temps de volatilité, PIT, risques concurrents
├── realtime.py      probabilité de survie, barrières à deux côtés, calibration
├── levels.py        niveaux de confluence + placebo à distance appariée
├── evaluate.py      walk-forward purgé, Brier, CRPS, DSR, bootstrap par blocs
└── selftest.py      21 vérifications Monte-Carlo des formules fermées

research/          Pipeline empirique (télécharge des données publiques)
├── data.py          Yahoo Finance + validation de la reconstruction horaire
├── engine.py        étude + backtest walk-forward + Monte-Carlo
├── report.py        génération du PDF
└── out/             PDF, figures, résultats, blotter des trades

report/
└── HOD_LOD_rapport.md   Rapport théorique : modèles, équations, hypothèses, limites
```

### Reproduire

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m hodlod.selftest      # 21 auto-tests Monte-Carlo
.venv/bin/python -m research.data        # télécharge et valide les données
.venv/bin/python -m research.engine      # étude + backtest + Monte-Carlo
.venv/bin/python -m research.report      # régénère le PDF
```

Les données brutes ne sont pas versionnées (régénérables, et leur rediffusion
n'est pas notre objet) : `research.data` les retélécharge et les met en cache
dans `research/cache/`. Les **résultats**, eux, sont versionnés dans
`research/out/` — un clone suffit pour lire le PDF sans rien exécuter.

### Méthode

Aucune fuite d'information : `σ` est un HAR en fenêtre expansive réestimé tous
les 20 jours et n'utilise que les variances ≤ J−1 ; le backtest entre à
l'ouverture de la barre suivant le signal ; le seuil et le take-profit sont
choisis par walk-forward mensuel sur le passé uniquement. Contrôles de
sur-ajustement : registre des essais, deflated Sharpe, test de permutation,
bootstrap par blocs, cohérence du signe entre instruments.
