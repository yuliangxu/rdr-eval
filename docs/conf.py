from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "RDR Eval"
author = "Yuliang Xu, Yun Wei, and Li Ma"
copyright = "2026, Yuliang Xu, Yun Wei, and Li Ma"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_title = "RDR Eval"
html_static_path = []

sphinx_gallery_conf = {
    "examples_dirs": str(ROOT / "examples"),
    "gallery_dirs": str(Path(__file__).parent / "auto_examples"),
    "filename_pattern": r"plot_",
    "ignore_pattern": r"(basic_usage|gaussian_1d)\.py",
    "download_all_examples": True,
    "within_subsection_order": "FileNameSortKey",
}
