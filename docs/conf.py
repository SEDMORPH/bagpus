# Sphinx configuration for bagpus documentation (readthedocs)

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'bagpus'
author = 'Vivienne Wild'
copyright = '2026, Vivienne Wild'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_nb',
]

# render the example notebooks without executing them (they need survey data
# and hours of compute); committed outputs are shown as-is
nb_execution_mode = 'off'

myst_enable_extensions = ['dollarmath', 'amsmath']

templates_path = ['_templates']
exclude_patterns = ['_build']

html_theme = 'sphinx_book_theme'
html_title = 'bagpus'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = '_static/bagpus_logo3.png'

# heavy dependencies not needed to build the docs
autodoc_mock_imports = ['bagpipes', 'torch', 'sbi', 'fast_histogram', 'joblib',
                        'sklearn', 'scipy', 'astropy', 'matplotlib', 'numpy']
