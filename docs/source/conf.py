import os
import sys

# Add the project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath('../../'))

project = 'Gossamer Threaded Intelligence'
author = 'Arboria Research'
release = '0.1.0'

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

# Ignore patterns
exclude_patterns = []

# HTML output theme
html_theme = 'alabaster'