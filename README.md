## Conda Env

Create and start the conda environment:

```sh
conda env create -f environment.yml 
conda activate xai-venv
```

## Installing
After installing you should be able to `import peepholelib` and its submodules

```sh
pip install .
```

## Developing

Add the following line at the beginning of your scrip to be able to import the library without installing it.
This is useful if your are developing code for the library or using a shared `venv`.

```sh
import sys
sys.path.insert(0, '<cloning path>/peepholelib')
```
