# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'SpatialEx'
copyright = '2025, Yonghao Liu, Chuyao Wang, and Xindi Dai'
author = 'Yonghao Liu, Chuyao Wang, and Xindi Dai'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', # 支持使用 Google 风格的 docstrings
    'sphinx.ext.viewcode', # 显示 [source] 链接
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    # 'sphinx_autodoc_typehints', # 把类型注解展示到文档里
    'nbsphinx'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # matplotlib 不用 GUI
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")  # 如有 numba，避免编译开销

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


autosummary_generate = True      # 让 autosummary 自动生成页面
napoleon_google_docstring = True
napoleon_numpy_docstring = False # 你用 Google 风格就把 NumPy 关掉即可
napoleon_include_init_with_doc = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
    "special-members": "__init__",
}

autodoc_mock_imports = [
    # 第三方大件
    "torch", "torchvision", "torchaudio","transformers",
    "numpy", "pandas", "tqdm", "matplotlib",
    "cellpose", "cellposesam", "cupy", "numba",
    # 包内的子模块（如果你只想展示一个 .py，就 mock 其他模块）
    "spatialex.model", "spatialex.utils", "spatialex.preprocess",
]
# --- 不要执行 .ipynb（无论你用 nbsphinx 还是 myst-nb）---
# nbsphinx
nbsphinx_execute = "never"
# myst-nb
nb_execution_mode = "off"
# nb_execution_mode = "off"         # 新版
# myst_nb_execute_notebooks = "off"  # 旧版


# autodoc_typehints = "description"  # 类型注解放到参数表里
# 让 Sphinx 找到你的包（按你的真实包名与路径调整）
import os, sys
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath("../../"))
# 添加spatialex包目录到Python路径
sys.path.insert(0, os.path.abspath("../../spatialex/"))
# --- 路径设置：把包含“spatialex/”的目录加到 sys.path ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]  # docs/source/conf.py -> 仓库根
sys.path.insert(0, str(ROOT))

# --- 可选：构建时快速自检（失败就直接报错，方便定位）---
try:
    import spatialex
    import spatialex.SpatialEx_pyG
    print("Import OK: spatialex.SpatialEx_pyG")
except Exception as e:
    print("WARN: import check failed but may be OK due to mocks:", e)
