from setuptools import setup

setup(
    name='peepholelib',
    version='0.0.0',
    description='Peepholes extraction',
    license='MIT',
    packages=['peepholelib', 'peepholelib/datasets', 'peepholelib/models', 'peepholelib/coreVectors', 'peepholelib/classifier', 'peepholelib/peepholes', 'peepholelib/utils', 'peepholelib/adv_atk', 'peepholelib/dummy'],
    author='Leandro de Souza Rosa',
    author_email='leandro.desouzarosa@unibo.it',
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
