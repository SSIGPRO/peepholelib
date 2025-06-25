import torch
from matplotlib import pyplot as plt
import seaborn as sb
from pandas import DataFrame as DF

if __name__ == '__main__':
    ds = 100
    d1 = torch.rand(ds)
    d2 = torch.rand(ds//2)
    df = DF({
        'Value': torch.hstack((d1, d2)),
        'Score': ['d1' for i in range(len(d1))]+['d2' for i in range(len(d2))],
        })

    colors = ['xkcd:dark cyan', 'xkcd:dark cyan']
    lines = ['--', '-']
    p = sb.kdeplot(data=df, x='Value', hue='Score', common_norm=False, palette=colors, clip=[0., 1.])
    
    handles = p.legend_.legend_handles[::-1]
    for ls, line, h in zip(lines, p.lines, handles):
        line.set_linestyle(ls)
        h.set_ls(ls)

    plt.show()

