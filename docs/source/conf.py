# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

def skip_undoc(app, what, name, obj, skip, options):
    if not skip:
        return obj.__doc__ in (None, "", " ", "\n")
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_undoc)


project = 'SigML'
copyright = '2025, Rishi Rao'
author = 'Rishi Rao'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    #'sphinx.ext.imgmath'
    'sphinx.ext.mathjax',
]

autodoc_mock_imports = ["torch","e3nn","numpy","scipy","nequip","ase","pymatgen","matplotlib","torch_geometric", "torch_cluster","torch_scatter","triqs","tqdm"]


templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
