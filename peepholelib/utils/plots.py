# General pytho stuff
from pathlib import Path as Path
from math import ceil

# plotting stuff
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd

# torch stuff
import torch
from torcheval.metrics import BinaryAUROC as AUC

def plot_ood(**kwargs):
    '''
    Plot OOD detection.

    Args:
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - id_loader (str): loaders of in-distribution data,  usually 'test'. Defaults to 'test'.
    - ood_loaders (list[str]): out-of-distribution loaders to consider

    - path ('str'): Path to save plots.
    - verbose (bool): print progress messages.
    '''
    scores = kwargs.get('scores')
    id_loader = kwargs.get('id_loader', 'test')
    ood_loaders = kwargs.get('ood_loaders')
    path = kwargs.get('path', None)
    verbose = kwargs.get('verbose', False)

    # parse arguments
    if path == None: 
        path = Path.cwd()
    else:
        path = Path(path)

    fig, axs = plt.subplots(1, len(ood_loaders), sharex='row', sharey='none', figsize=(4*len(ood_loaders), 4))
    
    for loader_n, ds_key in enumerate(ood_loaders):
        df_idood = pd.DataFrame()

        # save AUCs for each score type
        aucs = {}
        for score_name in scores[ds_key].keys():
            s_id = scores[id_loader][score_name] # TODO: rearragen iteration order
            s_ood = scores[ds_key][score_name]

            # computing AUC for each score type
            _labels = torch.hstack((torch.ones(s_id.shape), torch.zeros(s_ood.shape)))
            _scores = torch.hstack((s_id, s_ood))
            print('shapes: ', _labels.shape, s_id.shape, s_ood.shape)
            aucs[score_name] = AUC().update(_scores, _labels).compute().item()
            if verbose: print(f'AUC for {ds_key} {score_name} split: {aucs[score_name]:.4f}')

            df_idood = df_idood._append(
                    pd.DataFrame({
                        'score value': _scores,
                        'score type': \
                                [score_name+' ID' for i in range(len(s_id))] + \
                                [score_name+' OOD' for i in range(len(s_ood))]
                        }),
                    ignore_index = True,
                    )
        #--------------------
        # Plotting
        #--------------------
        # plotting IDs and OODs distribution
        # TODO: paremeterize
        colors = ['xkcd:cobalt', 'xkcd:cobalt', 'xkcd:bluish green', 'xkcd:bluish green']
        lines = ['--', '-', '--', '-']
                                                                                          
        ax = axs[loader_n] 
        p = sb.kdeplot(
                data = df_idood,
                ax = ax,
                hue = 'score type',
                x = 'score value',
                common_norm = False,
                palette = colors,
                clip = [0., 1.],
                alpha = 0.75,
                legend = loader_n == 0
                )
                                                                                          
        # set up linestyles
        for ls, line in zip(lines, p.lines):
            line.set_linestyle(ls)
        
        # set legend linestyle
        if loader_n == 0:
            handles = p.legend_.legend_handles[::-1]
            for ls, h in zip(lines, handles):
                h.set_ls(ls)
                                                                                          
        ax.set_xlabel('Score')
        ax.set_ylabel('%')
        title = f'{ds_key}'
        for score_name in aucs:
            title += f'\n{score_name} AUC={aucs[score_name]:.4f}'
        ax.title.set_text(title)
        ax.grid(True)

    plt.savefig((path/f'in_out_distribution.png').as_posix(), dpi=300, bbox_inches='tight')
    plt.close()
    return 

def plot_confidence(**kwargs):
    '''
    Plot OKs and KOs distributions and confidences. Confidences are computed for a score threshold 'th', assuming values bellow 'th' are wrongly classified and above are correctly classified true positive and negative ('TP(th)' and 'TN(th)') are computed, so 'conf(th) = (TP(th)+TN(th))/ns'. 'th' is plotted from 0 to 'max_score'. 

    Args:
    - corevectors (peepholelib.coreVectors.CoreVectors): corevectors with dataset parsed (see `peepholelib.coreVectors.parse_ds`).
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'scores'. Defaults to 'None'.
    - path ('str'): Path to save plots.
    - max_score (float): Max score for the accuracy plot, within '[0., 1.]'.
    - verbose (bool): print progress messages.
    '''

    cvs = kwargs.get('corevectors')
    scores = kwargs.get('scores')
    loaders = kwargs.get('loaders', None)
    max_score = kwargs.get('max_score', 20)
    calib_bin = kwargs.get('calib_bin', 1)
    path = kwargs.get('path', None)
    verbose = kwargs.get('verbose', False)

    # parse arguments
    if path == None: 
        path = Path.cwd()
    else:
        path = Path(path)

    if loaders == None: loaders = list(scores.keys())

    fig, axs = plt.subplots(2, len(loaders), sharex='row', sharey='none', figsize=(4*len(loaders), 4*2))

    for loader_n, ds_key in enumerate(loaders):
        df_okko = pd.DataFrame()
        df_conf = pd.DataFrame()
        
        # save AUCs for each score type
        aucs = {}
        for score_name in scores[ds_key].keys():
            _scores = scores[ds_key][score_name]
            results = cvs._dss[ds_key]['result'] 
            ns = _scores.shape[0] # number of samples

            s_oks = _scores[results == True]
            s_kos = _scores[results == False]
        
            # compute AUC for score and model
            aucs[score_name] = AUC().update(_scores, results.int()).compute().item()
            if verbose: print(f'AUC for {ds_key} {score_name} split: {aucs[score_name]:.4f}')
        
            df_okko = df_okko._append(
                    pd.DataFrame({
                        'score value': torch.hstack((s_oks, s_kos)),
                        'score type': \
                                [score_name+': OK' for i in range(len(s_oks))] + \
                                [score_name+': KO' for i in range(len(s_kos))]
                                }),
                    ignore_index = True,
                    )
            
            # Compute accuracies
            s_acc = torch.zeros(int(100*max_score)+1)
            for i, th in enumerate(torch.linspace(0., max_score, int(100*max_score)+1)):
                s_idx = _scores >= th 
                s_acc[i] = (results[s_idx].sum() + results[s_idx.logical_not()].logical_not().sum())/ns

            df_conf = df_conf._append(
                    pd.DataFrame({
                        'conf value': s_acc,
                        'score type':[score_name for i in range(len(s_acc))],
                        }),
                    ignore_index = True,
                    )

        #--------------------
        # Plotting
        #--------------------

        # plotting OKs and KOs distribution
        # TODO: paremeterize
        colors = ['xkcd:cobalt', 'xkcd:cobalt', 'xkcd:bluish green', 'xkcd:bluish green']
        lines = ['--', '-', '--', '-']

        ax = axs[0][loader_n] 
        p = sb.kdeplot(
                data = df_okko,
                ax = ax,
                hue = 'score type',
                x = 'score value',
                common_norm = False,
                palette = colors,
                clip = [0., 1.],
                alpha = 0.75,
                legend = loader_n == 0
                )

        # set up linestyles
        for ls, line in zip(lines, p.lines):
            line.set_linestyle(ls)
        
        # set legend linestyle
        if loader_n == 0:
            handles = p.legend_.legend_handles[::-1]
            for ls, h in zip(lines, handles):
                h.set_ls(ls)

        ax.set_xlabel('Score')
        ax.set_ylabel('%')
        title = f'{ds_key}'
        for score_name in aucs:
            title += f'\n{score_name} AUC={aucs[score_name]:.4f}'
        ax.title.set_text(title)
        ax.grid(True)

        # plotting Confidences
        # TODO: paremeterize
        colors = ['xkcd:cobalt', 'xkcd:bluish green']
        ax = axs[1][loader_n]
        sb.lineplot(
                data = df_conf,
                ax = ax,
                x = torch.linspace(0., max_score, int(100*max_score)+1).repeat(len(scores[ds_key].keys())),
                y = 'conf value',
                hue = 'score type',
                palette = colors,
                alpha = 0.75,
                legend = loader_n == 0,
                )
        ax.set_xlabel('% dropped')
        ax.set_ylabel('Accuracy (%)')
        ax.grid(True)

    plt.savefig((path/f'confidence.png').as_posix(), dpi=300, bbox_inches='tight')
    plt.close()
    return 

def plot_calibration(**kwargs):
    '''
    Plot calibration curves.

    Args:
    - corevectors (peepholelib.coreVectors.CoreVectors): corevectors with dataset parsed (see `peepholelib.coreVectors.parse_ds`).
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'scores'. Defaults to 'None'.
    - path ('str'): Path to save plots.
    - calib_bin (int): Bin size for calibration plot.
    - verbose (bool): print progress messages.
    '''

    cvs = kwargs.get('corevectors')
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
    print(path)

    if loaders == None: loaders = list(scores.keys())

    fig, axs = plt.subplots(1, len(loaders), sharex='none', sharey='none', figsize=(4*len(loaders), 4))

    n_bins = ceil(1/calib_bin)
    for loader_n, ds_key in enumerate(loaders):
        df_calib = pd.DataFrame()

        eces = {}
        for score_name in scores[ds_key].keys():
            _scores = scores[ds_key][score_name]
            results = cvs._dss[ds_key]['result'] 
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
            eces[score_name] = (s_ns*(s_acc-s_conf).abs()).sum()/ns

            df_calib = df_calib._append(
                    pd.DataFrame({
                        'accuracy': s_acc,
                        'score type': [score_name for i in range(n_bins)]
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
        # TODO: paremeterize
        colors = ['xkcd:cobalt', 'xkcd:bluish green']
        ax = axs[loader_n]
        _ax = sb.barplot(
                data = df_calib,
                ax = ax,
                x = x.repeat(len(scores[ds_key].keys())),
                y = 'accuracy',
                hue = 'score type',
                dodge = False,
                palette = colors,
                alpha = 0.75,
                width = 1.,
                legend = loader_n == 0,
                )

        sb.pointplot(
                data = df_perf_calib,
                ax = ax,
                x = 'x',
                y = 'y',
                markersize = 0,
                hue = 'score type',
                palette = ['xkcd:brick red'],
                alpha = 0.75,
                legend = loader_n == 0,
                )

        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy (%)')
        title = f'{ds_key}'
        for score_name in eces:
            title += f'\n{score_name} ECE={eces[score_name]:.2f}' 
        ax.title.set_text(title)
        ax.grid(True)


    plt.savefig((path/f'calibration.png').as_posix(), dpi=300, bbox_inches='tight')
    plt.close()
    return 
