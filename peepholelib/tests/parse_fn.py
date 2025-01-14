from peepholes.peepholes import peep_matrices_from_svds as pmfs
from peepholes.peepholes import Peepholes
import numpy as np
from numpy.random import rand, randint

if __name__=="__main__":
    ph = Peepholes()
    d = {
            'banana':{'S':rand(3,3), 'aaa':rand(3,3), 'Vh':rand(3, 3)},
            'potato':{'S':rand(3,4), 'aaa':rand(3,4), 'Vh':rand(3, 4)},
            'abacate':{'S':rand(3,5), 'aaa':rand(3,5), 'Vh':rand(3, 5)}
        }
    print('data: ', d)
    
    ranks = dict()
    for lk in d:
        ranks[lk]=randint(1,3) 
    print(ranks)
    parser_kwargs = {'rank': ranks}
    ph.get_peepholes(model=None,
                     activations=None,
                     path='path',
                     name='name',
                     peep_matrices=d,
                     parser=pmfs,
                     parser_kwargs=parser_kwargs
                     )
