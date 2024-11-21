# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import toml

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


@dataclass
class PoetryConfig:
    project: str
    version: str
    author: str


def read_poetry_config(filename: Path | str) -> PoetryConfig:
    filename = Path(filename)
    with open(filename, "r") as file:
        config = toml.load(file)["project"]
    return PoetryConfig(project=config["name"], version=config["version"], author=config["authors"][0]["name"])


poetry_config = read_poetry_config(project_root / "pyproject.toml")
project = poetry_config.project
author = poetry_config.author
copyright = f"{datetime.now().year}, {author}"
version = poetry_config.version


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
