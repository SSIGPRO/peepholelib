# General pytho stuff
from pathlib import Path as Path
from math import ceil

# plotting stuff
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sb
import pandas as pd

# torch stuff
import torch

def plot_calibration(**kwargs):
    '''
    Plot calibration curves.

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'scores'. Defaults to 'None'.
    - path ('str'): Path to save plots.
    - calib_bin (int): Bin size for calibration plot.
    - verbose (bool): print progress messages.
    '''
    
    dss = kwargs.get('datasets')
    scores = kwargs.get('scores')
    loaders = kwargs.get('loaders', None)
    calib_bin = kwargs.get('calib_bin', 0.1)
    path = kwargs.get('path', None)
    verbose = kwargs.get('verbose', False)

    # parse arguments
    if path == None: 
        path = Path.cwd()
    else:
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if loaders == None: loaders = list(scores.keys())

    fig, axs = plt.subplots(1, len(loaders), sharex='none', sharey='none', figsize=(5*(len(loaders)), 5))

    colors = ['xkcd:cobalt', 'xkcd:bluish green', 'xkcd:light orange', 'xkcd:dark hot pink', 'xkcd:purplish']

    n_bins = ceil(1/calib_bin)
    for loader_n, ds_key in enumerate(loaders):

        df_calib = pd.DataFrame()
        for score_name in scores[ds_key].keys():
            _scores = scores[ds_key][score_name]
            results = dss._dss[ds_key]['result'] 
            ns = _scores.shape[0] # number of samples

            # comput calibration and ECEs
            s_acc = torch.zeros(n_bins)
            s_conf = torch.zeros(n_bins)
            s_ns  = torch.zeros(n_bins)

            for b in range(n_bins):
                s_idx = torch.logical_and(_scores>b*calib_bin, _scores<=(b+1)*calib_bin)
                s_ns[b] = s_idx.sum() # how many samples in the bin
                s_acc[b] = (results[s_idx]).sum()/s_ns[b] # bin's accuracy
                s_conf[b] = _scores[s_idx].sum()/s_ns[b] # bin's average confidence

            # avoid numerical problems with NaN
            s_acc = torch.nan_to_num(s_acc) 
            s_conf = torch.nan_to_num(s_conf) 

            # Compute ECE score
            ece = ((s_ns*(s_acc-s_conf).abs()).sum()/ns).item()
            if verbose: print(f'ECE for {ds_key} {score_name} split: {ece:.4f}')

            df_calib = df_calib._append(
                    pd.DataFrame({
                        'accuracy': s_acc,
                        'score type': [score_name+f': ECE={ece:.2f}' for i in range(n_bins)]
                        }),
                    ignore_index = True,
                    )
        
        # add perfect calibration
        x = torch.linspace(0, 1, n_bins+1)[:-1]
        df_perf_calib = pd.DataFrame({
            'x': x,
            'y': x, 
            'score type': ['Perfect Calibration' for i in range(n_bins)]
            })

        #--------------------
        # Plotting
        #--------------------
        ax = axs[loader_n]
        sb.pointplot(
                data = df_calib,
                ax = ax,
                x = x.repeat(len(scores[ds_key].keys())),
                y = 'accuracy',
                hue = 'score type',
                palette = colors[0:len(scores[loaders[-1]])],
                alpha = 0.75,
                markersize = 8,
                legend = True
                )

        sb.pointplot(
                data = df_perf_calib,
                ax = ax,
                x = 'x',
                y = 'y',
                markersize = 0,
                hue = 'score type',
                palette = ['xkcd:red'],
                alpha = 0.75,
                legend = loader_n == len(loaders)-1,
                )

        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy (%)')
        ax.title.set_text(f'{ds_key}')
        ax.grid(True)
    
    plt.savefig((path/f'calibration.png').as_posix(), dpi=300, bbox_inches='tight')
    plt.close()
    return 
