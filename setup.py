from setuptools import setup

setup(
    name='peepholelib',
    version='0.0.0',
    description='Peepholes extraction',
    license='MIT',
    packages=[
        'peepholelib', 
        'peepholelib/adv_atk', 
        'peepholelib/coreVectors', 
        'peepholelib/datasets', 
        'peepholelib/featureSqueezing', 
        'peepholelib/models', 
        'peepholelib/peepholes', 
        'peepholelib/plots',
        'peepholelib/scores',
        'peepholelib/training',
        'peepholelib/utils',
    ],
    author=['Leandro de Souza Rosa', 'Lorenzo Capelli'],
    author_email=['leandro.desouzarosa@unibo.it','l.capelli@unibo.it'],
    keywords=['explainable AI, Attack detection, Confidence Estimation'],
    url='https://github.com/SSIGPRO/XAI',
    install_requires=[
        'numpy',
        'torch',
        'tensordict',
        'torchvision',
        'torchgmm',
        'seaborn',
        'cuda_selector',
      ],
)
