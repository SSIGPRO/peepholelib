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

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",  # or "lualatex" / "xelatex"
    "text.usetex": True,
    "font.family": "serif",
})

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

    colors = ['xkcd:cobalt', 'xkcd:bluish green', 'xkcd:light orange', 'xkcd:dark hot pink', 'xkcd:purplish', 'xkcd:slate gray', 'xkcd:cinnamon']

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
                        # 'score type': [score_name+f': ECE={ece:.2f}' for i in range(n_bins)]
                        'score type': [score_name+f': {ece:.2f}' for i in range(n_bins)]
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
        if len(loaders) == 1:
            ax = axs
        else:
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
                linestyles=["--"],
                palette = ['xkcd:red'],
                alpha = 0.6,
                # legend = loader_n == len(loaders)-1,
                )
        
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_xlabel('Confidence', fontsize=18,)
        ax.set_ylabel('Accuracy (%)', fontsize=18,)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        # ax.title.set_text(f'{ds_key}')
        ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        labels = [lab.replace("DMD-in", "DMD-a") for lab in labels]
        labels = [lab.replace("DMD-B", "DMD-b") for lab in labels]
        keys = list(scores[ds_key].keys())

        custom_order = [k for k in keys if k not in ['DMD-in', 'DMD-B']]
        # labels = list(labels)

        # optional: remove duplicates (Seaborn often repeats them)
        # by_label = dict(zip(labels, handles))
        # handles = list(by_label.values())
        # labels  = list(by_label.keys())
        
        custom_order.append("DMD-b")
        custom_order.append("DMD-a")
        custom_order.append("Perfect Calibration")

        def sort_key(label):
            for i, name in enumerate(custom_order):
                if label.startswith(name):
                    return i
            return len(custom_order)

        handles_labels = sorted(zip(handles, labels), key=lambda x: sort_key(x[1]))
        handles, labels = zip(*handles_labels)
        handles = list(handles)
        labels  = list(labels)

        # optional: make labels "table-like" with monospace alignment
        # labels currently look like "LACS: 0.03", "MSP: 0.12", ...
        methods, scores_str = [], []
        for lab in labels:
            if ": " in lab:
                m, s = lab.split(": ")
                methods.append(m)
                scores_str.append(s)
            else:
                methods.append(lab)
                scores_str.append("")

        labels = [f"{m:<8} {s:>5}" for m, s in zip(methods, scores_str)]

        ax.set_xlabel('Confidence', fontsize=18)
        ax.set_ylabel('Accuracy (%)', fontsize=18)
        ax.grid(True)

        ax.legend(
                handles,
                labels,
                title="ECE Scores",
                loc="upper center",
                bbox_to_anchor=(0.5, 1.32),   # move legend above axes
                ncol=4,                       # <-- split into 4 columns
                # mode="expand",                # makes columns spread nicely
                prop={"family": "monospace"},
                fontsize=24,
                title_fontsize=18,
                # borderpad=1.0,
                # labelspacing=0.6
            )

        # ax.legend(
        #     handles,
        #     labels,
        #     title="ECE Scores",
        #     loc="center left",
        #     bbox_to_anchor=(1.05, 0.5),
        #     prop={"family": "monospace"},  
        #     fontsize=50,
        #     title_fontsize=20,    
        #     # markerscale=1.8,      # bigger symbols
        #     #handlelength=3,       # longer line segments
        #     borderpad=1.2,        # padding inside the box
        #     labelspacing=0.8, 
        # )
    
    plt.savefig((path/f'calibration.png').as_posix(), dpi=300, bbox_inches='tight')
    plt.savefig((path / "calibration.pgf").as_posix())
    plt.close()
    return 
