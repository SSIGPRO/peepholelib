import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:  # seaborn is optional, fall back to matplotlib styling
    sns = None

import os
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import math
from torch import nn

import tqdm
sys.path.append('/home/francescoaldrigo/SPACE/francescoaldrigo')


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0.00005):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def load_and_process_data(data_dir,dataset_name, window_size=16, col_to_remove=None, flag_col = "ACT40082(PREPRO_CMG_ENABLE)", scaler = StandardScaler()):
    
    df = pd.read_csv(os.path.join(data_dir, dataset_name))
    df = df.drop(columns =[ c for c in df.columns if c.startswith("ANT47") or c.startswith("Frame")])

    # Creiamo un dizionario di Series senza NaN
    series_dict = {col: df[col] for col in df.columns}
    # Troviamo la lunghezza minima tra tutte le Series
    min_len = min(len(s) for s in series_dict.values())
    # Allineiamo tutte le Series alla lunghezza minima
    aligned_data = {col: s.iloc[:min_len].reset_index(drop=True) for col, s in series_dict.items()}
    # Creiamo un nuovo DataFrame con i dati allineati
    df_aligned = pd.DataFrame(aligned_data)
    print(df_aligned.head())
    print("Lunghezza del nuovo dataset:", len(df_aligned))

    # colonna flag operativo
    #flag_col = "ACT40082(PREPRO_CMG_ENABLE)"

    # colonne da rimuovere (esempio: digitali o flag)
    # taglio dataset da quel punto
    
    cols_to_drop = col_to_remove  # aggiungi altre colonne se vuoi
    # primo cambio di stato
    changes = df_aligned[flag_col].diff().fillna(0)
    first_change_idx = changes[changes != 0].index[0]
    df_start = df_aligned.loc[first_change_idx:].reset_index(drop=True)
    # nuovo dataset senza queste colonne
    data_ready = df_start.drop(columns=col_to_remove)

    # colonne rimanenti da plottare
    plot_cols = data_ready.select_dtypes(include='number').columns.tolist()
    
    num_cols = len(plot_cols)

    scaled = scaler.fit(data_ready)
    
    data_norm = pd.DataFrame(
    data=scaled.transform(data_ready),
    index=data_ready.index,
    columns=data_ready.columns,
    )
   # Get the original shape
    rows = data_norm.shape[0]

    # Trim the number of rows to be an exact multiple of the window size
    trimmed_rows = (rows // window_size) * window_size
    df_trimmed = data_norm.iloc[:trimmed_rows]

    # Convert to numpy array and reshape into windows
    data_grouped = df_trimmed.to_numpy().reshape(-1, window_size, num_cols)

    # Remove windows that contain any NaN values
    mask = ~np.isnan(data_grouped).any(axis=(1, 2))
    data_cleaned = data_grouped[mask]

    

    # Debug prints
    print(f"Selected channels: {num_cols}")
    print(f"Shape after reshaping: {data_grouped.shape}")
    print(f"Shape after removing NaNs: {data_cleaned.shape}")

    return data_cleaned,num_cols

class ModelConfig:
    def __init__(self):
        self.architecture = "conv_ae1D"  # Tipo di modello
        self.kernel_size = 5  # Dimensione kernel convoluzione
        self.filter_num = 42  # Numero di filtri nelle convoluzioni
        self.stride = 2  # Passo della convoluzione
        self.pool = 0  # Fattore di pooling
        self.latent_dim = 100  # Dimensione dello spazio latente
        self.lay3 = True  # Numero di layer
        self.activation = nn.ELU(alpha=1)  # Funzione di attivazione
        self.bn = True  # Batch Normalization
        #self.increasing = 0  # Flag per determinare la crescita dei filtri
        #self.flattened = 0  # Indica se il modello è appiattito
        self.dilation = 1  # Dilation rate per convoluzioni
        self.padding = 4
        self.in_channel = 16
        

class DatasetConfig:
    def __init__(self):
        self.n_features = 16  # Numero di feature nel dataset
        self.sequence_length = 16  # Lunghezza della sequenza temporale (seq_in_length)
        self.batch_size = 500  # Dimensione del batch
        self.columns_subset = 16  # Numero di colonne da usare
        self.dataset_subset = None  # Numero di righe da usare
        self.shuffle = True  # Indica se mescolare i dati
        self.train_val_split = 0.8  # Percentuale di dati per il training

class OptimizerConfig:
    def __init__(self):
        self.lr = 0.003  # Learning rate
        self.lr_patience = 5  # Pazienza del learning rate scheduler
        self.epochs = 200  # Numero di epoche di training
        self.es_patience = 10  # Early stopping patience

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.dataset = DatasetConfig()
        self.opt = OptimizerConfig()

# Creazione dell'istanza di configurazione
cfg = Config()

def plot_all_channels_comparison(original_data, corrupted_data, feature_names,
                                 corrupted_feature, anomaly_name, window_idx=0,
                                 n_cols=4, figsize=(15, 3)):
    """
    Confronta visivamente tutti i canali per una finestra specifica,
    adattando la griglia al numero di feature.

    Parameters:
    - original_data: array 3D (N_windows, seq_len, n_features)
    - corrupted_data: array 3D (N_windows, seq_len, n_features)
    - feature_names: lista dei nomi delle feature
    - corrupted_feature: nome del canale corrotto
    - anomaly_name: nome dell'anomalia applicata
    - window_idx: indice della finestra da visualizzare
    - n_cols: numero di colonne nei subplot (default 4)
    - figsize: (width, height_per_row) → altezza per riga
    """
    if sns is not None:
        sns.set_style("whitegrid")
    else:
        plt.style.use("default")
    plt.rcParams.update({'font.size': 8})

    num_features = len(feature_names)
    n_rows = math.ceil(num_features / n_cols)
    fig_w, h_per_row = figsize

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_w, h_per_row * n_rows), sharex=True)
    axs = axs.flatten()

    corrupted_idx = feature_names.index(corrupted_feature)

    for i, fname in enumerate(feature_names):
        ax = axs[i]
        ax.plot(original_data[window_idx, :, i], label="Original", color='gray', linewidth=1.5)
        ax.plot(corrupted_data[window_idx, :, i], label="Corrupted", color='crimson', linestyle='--', linewidth=1.2)

        if i == corrupted_idx:
            ax.set_title(f"{fname} (Corrupted - {anomaly_name})", color='crimson', fontweight='bold', fontsize=9)
        else:
            ax.set_title(f"Channel: {fname}", fontsize=9)

        ax.grid(True, linestyle='--', alpha=0.4)

    # spegni eventuali subplot vuoti
    for j in range(num_features, len(axs)):
        axs[j].axis("off")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', ncol=2, fontsize=9)
    plt.suptitle(f"Original vs Corrupted signals\n(Anomaly: {anomaly_name}, Window: {window_idx})",
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()