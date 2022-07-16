# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../.."))
import deepdrivemd  # noqa


# -- Project information -----------------------------------------------------

project = "deepdrivemd"
author = "Alexander Brace, Hyungro Lee, Heng Ma, Anda Trifan, Matteo Turilli, Igor Yakushin, Li Tan, Andre Merzky, Tod Munson, Ian Foster, Shantenu Jha, Arvind Ramanathan"
now = datetime.datetime.now()
copyright = "2020-{}, ".format(now.year) + author

# The full version, including alpha/beta/rc tags
release = deepdrivemd.__version__
version = deepdrivemd.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinxcontrib.autodoc_pydantic",
]

# Autosummary settings
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Autodoc settings
# Need to figure these out. See https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_default_options
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# List of imports to mock when building the documentation.
autodoc_mock_imports = [
    "adios2",
    "tensorflow",
    "simtk.openmm",
    "openmm",
    "cupy",
    "cuml",
    "numba",
    "pandas",
    "h5py",
    "sklearn",
    "wandb",
    "torch",
    "torchsummary",
    "mdlearn",
    "mdtools",
    "molecules",
    "MDAnalysis",
]

html_context = {
    "conf_py_path": "/docs/source/",  # Path in the checkout to the docs root
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
