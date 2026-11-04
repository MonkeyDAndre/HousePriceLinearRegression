# House Price Linear Regression

Progetto del modulo *Fondamenti di Machine Learning* (ProfessionAI): stima dei prezzi delle case con la
regressione lineare multipla e confronto con i modelli regolarizzati Ridge, Lasso ed Elastic Net.

## La domanda
Il motore di valutazione di RealEstateAI Solutions usa una regressione lineare multipla che, sugli immobili
nuovi, sembra sbagliare più del previsto. La **regolarizzazione** migliora le stime? Quali caratteristiche
pesano davvero sul prezzo? Il modello sbaglia in modo sistematico su qualche fascia di prezzo?

## Cosa contiene la repository

| Percorso | Contenuto |
|---|---|
| `notebooks/Regressione_lineare_housing.ipynb` | Il notebook del progetto, con codice, grafici e commenti |
| `housing.csv` | Il dataset della consegna (545 case, 12 caratteristiche + prezzo) |
| `Housing_kaggle.csv` | Il dataset originale di Kaggle ([Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)), usato per verificare la codifica di `furnishingstatus` |
| `docs/superpowers/specs/` | Documento di design del progetto |
| `docs/superpowers/plans/` | Piano di implementazione del notebook |
| `requirements.txt` | Librerie Python necessarie |

## Come è costruita l'analisi
1. Pulizia del dataset ed esplorazione dei dati. Il confronto con il dataset di Kaggle mostra che nel CSV i
   codici 1 e 2 di `furnishingstatus` sono invertiti rispetto alla documentazione.
2. Split train/test stratificato per fasce di prezzo e regressione lineare come baseline.
3. Lineare, Ridge, Lasso ed Elastic Net con due codifiche di `furnishingstatus` (ordinal con l'ordine
   corretto e one-hot), sul prezzo e sul logaritmo del prezzo: 16 configurazioni, valutate con una
   cross-validation ripetuta 5 × 3 e con ricerca dell'iperparametro `alpha`.
4. Scelta del modello finale (RMSE di cross-validation più basso) e valutazione sul test set.

## Risultati principali
- La regressione lineare **non va in overfitting** su questi dati: l'errore su case nuove è uguale a quello
  in addestramento.
- **La regolarizzazione non migliora l'errore in modo misurabile**: tutte le 16 configurazioni hanno un RMSE
  di cross-validation tra 1,074 e 1,096 milioni, con differenze molto più piccole della variabilità tra fold
  (circa 0,11 M).
- Modello finale: **Ridge sul logaritmo del prezzo** con codifica one-hot. Sul test set ha RMSE 1,08 M e
  R² 0,68, praticamente come la regressione lineare semplice.
- Le caratteristiche più influenti sono aria condizionata, numero di bagni, strada principale, zona
  preferita, numero di piani e area.
- Tutti i modelli sottostimano le case più costose (di circa 1 milione in media): per ridurre l'errore
  servono soprattutto nuove informazioni sugli immobili. Modelli più complessi (es. random forest)
  potrebbero cogliere effetti non lineari, ma con poche feature e pochi dati il guadagno atteso è limitato.

## Come eseguire il notebook
Serve Python 3.11 o successivo.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (su macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook notebooks/Regressione_lineare_housing.ipynb
```

Il notebook legge il dataset direttamente dall'URL della consegna, quindi serve una connessione a
internet; `housing.csv` ne è una copia locale. L'esecuzione completa richiede una decina di minuti e usa
tutti i core della CPU.

## Licenza
Distribuito con licenza MIT: vedi [LICENSE](LICENSE).
