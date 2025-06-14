# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Segma Vision Pro Light'
copyright = '2025, Alae Boutarhat  & Salma Bourkiba'
author = 'Alae Boutarhat  & Salma Bourkiba'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']












# Thème et options
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': True,
    'style_nav_header_background': '#2980B9',
}

# CSS personnalisé
html_static_path = ['_static']
html_css_files = ['custom.css']




html_static_path = ['_static']
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 3,
    'titles_only': True,
    'collapse_navigation': False,
}



html_css_files = ['custom.css']
rst_prolog = '''
.. role:: blue-bold
   :class: blue-bold-term
'''
