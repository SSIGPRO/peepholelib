import seaborn as sb
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    a = np.random.normal(5, 1, 12183)
    ca = ['ok' for x in a]
    b = np.random.normal(2, 1, 2423)
    cb = ['ko' for x in b]
    df = DataFrame({'score':np.hstack((a, b)), 'class':np.hstack((ca, cb))})

    plt.figure()
    sb.histplot(data=df, bins=100, x='score', hue='class') 
    plt.savefig('test.png', dpi=300, bbox_inches='tight')
    plt.close()
