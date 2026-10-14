# Design: confronto tra modelli regolarizzati per la stima dei prezzi residenziali

**Data:** 2026-10-14
**Contesto:** progetto del modulo "Fondamenti di Machine Learning" (ProfessionAI). Richiesta in `Consegna/`.

## 1. Obiettivo

RealEstateAI Solutions stima i prezzi con una regressione lineare multipla che va in overfitting
sugli immobili nuovi. Si chiede un confronto onesto e documentato tra Ridge, Lasso ed Elastic Net,
partendo da una regressione lineare multipla come baseline, con:
preprocessing motivato, validazione corretta (k-fold CV), MSE (+ R²), lettura dei coefficienti
(quanti ≠ 0, quali più grandi), analisi dei residui (errori sistematici per fascia di prezzo),
grafici a supporto delle decisioni, notebook leggibile da non programmatori, conclusioni con
raccomandazione operativa.

## 2. Deliverable

| File | Scopo | Destinazione |
|---|---|---|
| `notebooks/01_confronto_modelli_regolarizzati.ipynb` | Consegna (livello "B") | Google Colab |
| `notebooks/02_analisi_estesa.ipynb` | Approfondimento personale (livello "C") | solo locale / repo |
| `requirements.txt` | Ambiente locale riproducibile | repo |

Vincoli comuni a entrambi i notebook:
- **Autosufficienti**: nessun modulo `.py` esterno, nessun path locale. Il dataset è letto da
  `https://proai-datasets.s3.eu-west-3.amazonaws.com/housing.csv`.
- **Eseguibili su Colab senza modifiche** (sviluppo in locale in `.venv`). Tutte le librerie usate
  sono preinstallate su Colab, quindi non serve alcuna cella `%pip install`.
- **Lingua**: italiano. Alternanza di celle markdown (cosa / perché / cosa osserviamo) e codice.
- **Riproducibilità**: `RANDOM_STATE = 42` usato ovunque.
- Ogni grafico ha titolo esplicativo, assi con unità di misura e una cella "Cosa osserviamo".

## 3. Fatti sul dataset (verificati)

- 545 righe, 13 colonne, **nessun valore mancante**.
- Colonne: `price` (target), `area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `guestroom`,
  `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, `furnishingstatus`.
- **Unità di misura originali** (dataset Kaggle "Housing Prices", mercato indiano):
  prezzo in **rupie indiane (INR)**, area in **piedi quadrati (sq ft)**. Si addestra sempre in
  unità originali. La conversione in euro avviene solo nella tabella degli effetti, dichiarando
  tasso e data; nota per il lettore che la conversione cambia l'unità, non il mercato.
- **Il CSV è ordinato per prezzo decrescente** → ogni split/CV deve usare `shuffle=True`.
- **`furnishingstatus` ha codifica invertita rispetto alla mail.** Conteggi CSV: 0=178, 1=140,
  2=227; conteggi Kaggle: unfurnished=178, furnished=140, semi-furnished=227. Quindi nel CSV
  **0 = non arredato, 1 = arredato, 2 = semi-arredato** (la mail dice 1 = parziale, 2 = completo).
  Va mostrato e motivato nella sezione 3 del notebook.
- Le altre 6 categoriche sono binarie 0/1: per esse ordinal e one-hot (`drop='first'`) sono
  identici per costruzione — da dichiarare.

## 4. Notebook 1 (consegna) — struttura narrativa

0. **Introduzione** (markdown): contesto, domanda, come leggere il notebook, indice.
1. **Librerie**: tutti gli import, `RANDOM_STATE`, stile grafici.
2. **Funzioni di supporto**, in tre celle ciascuna con markdown:
   - 2a. Preprocessing: costruzione delle tre varianti di encoding + scaling in `Pipeline`.
   - 2b. Addestramento/valutazione: GridSearchCV, metriche, conteggio coefficienti ≠ 0,
     conversione coefficienti in effetti reali.
   - 2c. Grafici (funzioni riutilizzabili per i grafici del §6).
3. **Lettura e pulizia**: caricamento da URL, tipi, mancanti, duplicati, unità di misura,
   scoperta e correzione della codifica di `furnishingstatus`.
4. **Analisi esplorativa** (grafici 1–5 del §6).
5. **Split train/test e baseline**: split stratificato; regressione lineare multipla, errore
   train vs test. Se l'overfitting non è evidente lo si dice apertamente e si motiva la
   regolarizzazione con stabilità dei coefficienti e selezione delle feature.
6. **Modelli regolarizzati × encoding**: LinearRegression, Ridge, Lasso, ElasticNet × 3 encoding
   (12 configurazioni), stesso protocollo (§5).
7. **Confronto e scelta encoding**: grafici 7–10, regola di scelta del §5.
8. **Passaggio al log del prezzo**: 4 modelli con l'encoding scelto e `log(price)` come target.
9. **Prezzo originale vs log**: grafici 11–12; gli effetti del modello log sono percentuali.
10. **Valutazione finale sul test set**: solo configurazioni finaliste, test set usato una volta;
    grafici 13–15.
11. **Conclusioni**: miglior modello, coefficienti ed effetti reali (grafico 16), performance in
    unità leggibili, limiti, raccomandazione operativa.

### Varianti di encoding (sezione 6)

| Nome | `furnishingstatus` | Binarie |
|---|---|---|
| `ordinal_ingenuo` | codici CSV così come sono (0, 1=arredato, 2=semi) — ordine sbagliato | 0/1 |
| `ordinal_corretto` | ricodificato 0=non arredato, 1=semi, 2=arredato | 0/1 |
| `onehot` | `OneHotEncoder(drop='first')`, riferimento = non arredato | 0/1 |

## 5. Protocollo di validazione

- **Split**: `train_test_split(test_size=0.2, shuffle=True, random_state=42)` stratificato su
  5 quantili di `price` (`pd.qcut`). 436 train / 109 test. Il test set si usa solo in sezione 10.
- **CV**: `RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)` sul solo train (shuffle
  implicito in RepeatedKFold). Risultati come media ± deviazione standard sulle 15 valutazioni.
- **Pipeline**: `encoding → StandardScaler (tutte le feature, binarie incluse) → modello`,
  ricalcolata su ogni fold (nessun leakage).
- **Iperparametri** (`GridSearchCV`, scoring `neg_mean_squared_error`):
  - Ridge: `alpha` logspace ampio (indicativamente 1e-3 → 1e4).
  - Lasso: `alpha` logspace su scala adatta al target (sul prezzo grezzo valori grandi,
    indicativamente 1e0 → 1e6; sul log valori piccoli, 1e-5 → 1e0).
  - ElasticNet: stessa griglia di alpha del Lasso × `l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99}`.
  - Griglie distinte per target originale e log. Controllo automatico con avviso se l'alpha
    ottimo cade al bordo della griglia (in tal caso si allarga la griglia).
- **Metriche** (sempre in rupie, anche per il modello log): MSE (richiesto, con commento
  sull'ordine di grandezza ~10¹²), RMSE, MAE, R².
- **Target log**: `TransformedTargetRegressor(func=np.log, inverse_func=np.exp)`. Nota: `exp`
  stima la mediana condizionata → leggera sottostima (approfondita in E6).
- **Regola di scelta** (dichiarata prima dei risultati): vince il minimo RMSE medio di CV; se la
  differenza con un'alternativa è minore di una deviazione standard, si considerano equivalenti e
  si sceglie la più semplice/interpretabile.
- **Limite dichiarato**: selezione di alpha e stima dell'errore sulla stessa CV → stime
  leggermente ottimistiche; mitigato dal test set finale, quantificato in E4 (nested CV).

### Lettura dei coefficienti

Tabella per ogni modello finale con, per ciascuna feature:
- coefficiente standardizzato β (per la classifica di importanza);
- **effetto reale** = β / dev.std della feature (dev.std calcolata sul train), espresso come:
  rupie per passaggio 0→1 (binarie), per unità in più (camere, bagni, piani, posti auto), per
  sq ft e per m² (1 m² = 10.7639 sq ft) per `area`, rispetto al livello di riferimento (one-hot);
- conversione in euro con tasso e data dichiarati;
- per il modello log: effetto percentuale = (exp(β / dev.std) − 1) × 100%.

## 6. Grafici

Esplorazione: (1) istogramma prezzo e log-prezzo; (2) distribuzioni numeriche e frequenze
categoriche; (3) heatmap di correlazione; (4) scatter prezzo vs area colorato per una binaria;
(5) boxplot prezzo per categorica.
Baseline: (6) barre errore train vs test.
Confronto: (7) barre raggruppate RMSE CV ± std (4 modelli × 3 encoding); (8) errore CV vs alpha
(scala log) con alpha scelto evidenziato; (9) percorso dei coefficienti del Lasso; (10) numero di
coefficienti ≠ 0 per modello ed encoding.
Log vs originale: (11) barre metriche; (12) residui vs predetti affiancati.
Valutazione finale: (13) learning curve dei finalisti; (14) pannello residui: residui vs
predetti, istogramma con normale, errore medio per quintile di prezzo; (15) predetto vs reale con
bisettrice.
Conclusioni: (16) barre orizzontali degli effetti in euro ordinati per importanza.

Una sola palette coerente in tutto il notebook.

## 7. Notebook 2 (personale) — esperimenti

Stessa struttura (librerie, funzioni copiate dal notebook 1, storia). Ogni esperimento ha
domanda, codice, verdetto; tutti con il protocollo del §5 e confronto con il miglior modello del
notebook 1.

- **E1 – Scala delle binarie**: standardizzare tutto vs solo numeriche vs Gelman (numeriche /
  2·std, binarie 0/1). Effetto su errore e feature azzerate dal Lasso.
- **E2 – Outlier**: rilevazione (IQR su prezzo e area, distanza di Cook, residui studentizzati);
  strategie: tenere / rimuovere dal solo train / `HuberRegressor`.
- **E3 – Feature engineering**: log(area), stanze totali, area per stanza, interazioni
  (area×prefarea, area×mainroad); `PolynomialFeatures(degree=2)` con e senza regolarizzazione
  (scenario in cui l'overfitting della regressione lineare è atteso).
- **E4 – Nested CV**: stima non distorta vs CV semplice del notebook 1.
- **E5 – Test sui residui** (statsmodels): Shapiro-Wilk, Jarque-Bera, Breusch-Pagan, VIF; modello
  su prezzo vs log.
- **E6 – Smearing di Duan** per la retrotrasformazione del log.
- **E7 – Intervalli di previsione**: bootstrap dei residui e/o regressione quantile; copertura
  empirica sul test set. Da affrontare per ultimo.
- **Sintesi**: tabella riassuntiva (esperimento, configurazione, RMSE CV ± std, verdetto).

## 8. Ambiente e repository

- `.venv` locale; `requirements.txt` con pandas, numpy, scikit-learn, matplotlib, seaborn,
  statsmodels, jupyter, ipykernel.
- Repo git locale (branch `main`), da collegare in seguito a un remote GitHub fornito dall'utente.
- Verifica: ogni notebook deve girare dall'inizio alla fine senza errori (`jupyter nbconvert
  --execute`) prima di considerarlo concluso.

## 9. Fuori scopo

Modelli non lineari (alberi, boosting), deployment, API, dashboard.
