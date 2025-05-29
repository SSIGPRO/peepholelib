import torch
import abc 
from functools import partial
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

class HeadSqueezingDetector(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.score_fn = kwargs['score_fn'] 
        
        self.output = None
        self.output_dict = {}

    

       