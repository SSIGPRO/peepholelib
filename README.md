## Conda Env

Create and start the conda environment:

```sh
    conda env create -f environment.yml 
    conda activate xai-venv
```

Export env:
```sh
    conda env export | grep -v "^prefix: " > environment.yml
```

Update env:
```sh
conda env update --file local.yml --prune
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

## Deep K-NN

Install FALCONN outside our repo. We will need to update the `pybind11` library used in `falconn` since their version is outdated.
```sh
    cd <somewhere you like>
    git clone https://github.com/FALCONN-LIB/FALCONN
    cd FALCONN/external
    rm -rf pybind11
    git clone https://github.com/pybind/pybind11
    cd ..
```

If any error occurs, try eiditing lines 53 and 56 from `Makefile` to the following two lines respectively
```sh
	cd $(PYTHON_PKG_DIR); python setup.py sdist; cd dist; tar -xf falconn*.tar.gz; cd falconn-*; python setup.py build
	cd $(PYTHON_PKG_DIR)/dist/falconn-*; python setup.py install
```

Now we can install it:
```sh
    make python_package_install 
```
