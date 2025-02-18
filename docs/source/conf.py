# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'SpatialGlue'
copyright = '2023, Yahui Long'
author = 'Yahui Long'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'nbsphinx'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

#import os
#import sys

#sys.path.append(os.path.abspath("/home/docs/checkouts/readthedocs.org/user_builds/spatialglue-tutorials/checkouts/latest/docs/source/index.rst"))

#from recommonmark.parser import CommonMarkParser
#source_parsers = {
#    '.md': CommonMarkParser,
#}

#master_doc = '/home/docs/checkouts/readthedocs.org/user_builds/spatialglue-tutorials/checkouts/latest/docs/source/index'

source_suffix = '.rst'

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
