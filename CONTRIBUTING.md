# Contributing to DeepDriveMD

If you are interested in contributing to DeepDriveMD, your contributions will fall into two categories:

1. You want to implement a new feature:
    - In general, we accept any features as long as they fit the scope of this package. If you are unsure about this or need help on the design/implementation of your feature, post about it in an issue.
2. You want to fix a bug:
    - Please post an issue using the Bug template which provides a clear and concise description of what the bug was.

Once you finish implementing a feature or bug-fix, please send a Pull Request to https://github.com/DeepDriveMD/DeepDriveMD-pipeline.

## Developing DeepDriveMD

To develop DeepDriveMD on your machine, please follow these instructions:


1. Clone a copy of DeepDriveMD from source:

```
git clone https://github.com/DeepDriveMD/DeepDriveMD-pipeline.git
cd DeepDriveMD-pipeline
```

2. If you already have a DeepDriveMD from source, update it:

```
git pull
```

3. Install DeepDriveMD in `develop` mode:

```
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade wheel pip
python -m pip install -r requirements_dev.txt
```

This mode will symlink the Python files from the current local source tree into the Python install.
Hence, if you modify a Python file, you do not need to reinstall DeepDriveMD again and again.

4. Ensure that you have a working deepdrivemd installation by running:
```
python -c "import deepdrivemd; print(deepdrivemd.__version__)"
```

5. Before pushing changes to the repository, please run the dev tools to enforce consistent style, sorting of imports, linting and type checking with mypy:
```
make
```


## Unit Testing

We are planning to add a test suite in a future release which uses pytest for unit testing.

## Building Documentation

To build the documentation:

1. [Build and install](#developing-deepdrivemd) DeepDriveMD from source.
2. The `requirements_dev.txt` contains all the dependencies needed to build the documentation.
3. Generate the documentation file via:
```
cd DeepDriveMD-pipeline/docs
make html
```
The docs are located in `DeepDriveMD-pipeline/docs/build/html/index.html`.

To view the docs run: `open DeepDriveMD-pipeline/docs/build/html/index.html`.

## Releasing to PyPI

To release a new version of deepdrivemd to PyPI:

1. Merge the `develop` branch into the `main` branch with an updated version number in [`deepdrivemd.__init__`](https://github.com/DeepDriveMD/DeepDriveMD-pipeline/blob/main/deepdrivemd/__init__.py).
2. Make a new release on GitHub with the tag and name equal to the version number.
3. [Build and install](#developing-deepdrivemd) deepdrivemd from source.
4. Run the following commands:
```
python setup.py sdist
twine upload dist/*
```
