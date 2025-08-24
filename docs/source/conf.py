# docs/source/conf.py
# -------------------
# Sphinx config for SpatialEx docs (fast & RTD-friendly)

# ---- Project info ----
project = "SpatialEx"
author = "Yonghao Liu, Chuyao Wang, and Xindi Dai"
copyright = "2025, " + author
release = "0.1"
version = "0.1.0"

# ---- Keep the build lightweight on RTD ----
import os
os.environ.setdefault("MPLBACKEND", "Agg")        # no GUI backend
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")   # avoid JIT compile

# ---- Extensions (only the essentials) ----
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",            # render notebooks, but do NOT execute them
]

# Do not execute notebooks (critical to avoid timeouts)
nbsphinx_execute = "never"
# (if you switch to myst-nb in future) nb_execution_mode = "off"

# Intersphinx (optional)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

# Autosummary / Napoleon / Autodoc
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
# NOTE: If you want type hints in parameter tables later:
# extensions.append("sphinx_autodoc_typehints")
# autodoc_typehints = "description"

# ---- Mock heavy deps & internal modules you won't import on RTD ----
autodoc_mock_imports = [
    # heavy third-party
    "torch", "torchvision", "torchaudio",
    "numpy", "pandas", "matplotlib", "tqdm",
    "transformers", "cellpose", "cupy", "numba","timm","scanpy","anndata","scipy","scikit-learn","scikit-misc",
    # your own submodules (if you only show this one .py)
    "spatialex.model", "spatialex.utils", "spatialex.preprocess",
]

# ---- HTML theme ----
html_theme = "sphinx_rtd_theme"

# ---- Source suffix ----
source_suffix = ".rst"
templates_path = ["_templates"]
epub_show_urls = "footnote"

# ---- sys.path: add the REPO ROOT (parent of 'spatialex/') ----
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]    # docs/source/ -> repo root
sys.path.insert(0, str(ROOT))

# quick import check (non-fatal if mocks kick in)
try:
    import spatialex
    import spatialex.SpatialEx_pyG
    print("Import OK: spatialex.SpatialEx_pyG (mocks may apply)")
except Exception as e:
    print("WARN import:", e)
