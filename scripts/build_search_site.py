#!/usr/bin/env python3
"""Build FWIGAN's static search documentation with Python 3.10+ only.

Run from any directory. No package installation, network access, or GPU is used.
Output is deterministic and uses relative asset paths for project GitHub Pages.
Existing nonempty destinations must carry this builder's ownership marker. The
builder overwrites only its named files and never deletes a directory tree.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import struct
import sys
from pathlib import Path
from string import Template
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "site"
REPOSITORY = "https://github.com/izzatum/FWIGAN"
DEFAULT_BASE_URL = "https://izzatum.github.io/FWIGAN/"
MARKER = ".fwigan-site-builder"
MARKER_CONTENT = "FWIGAN static documentation builder v1\n"
IMAGE_NAME = "marmousi-120epoch-velocity-models.png"
PAGE_TITLE = "FWIGAN: Full-Waveform Inversion with PyTorch"
PAGE_DESCRIPTION = (
    "FWIGAN research software for 2D acoustic full-waveform inversion with "
    "PyTorch and DeepWave. Installation, GPU guidance, Marmousi results, and citation."
)
DEFAULT_DESCRIPTION = (
    "A research toolkit for 2D acoustic full-waveform inversion (FWI) and "
    "adversarial FWIGAN using DeepWave and PyTorch, with shared configuration, "
    "evaluation, and checkpoint/resume workflows."
)


def base_url(value: str) -> str:
    """Accept a complete http(s) site URL, preserving a project path prefix."""
    parts = urlsplit(value)
    if (
        parts.scheme not in {"https", "http"}
        or not parts.netloc
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or any(character.isspace() for character in value)
        or any(segment in {".", ".."} for segment in parts.path.split("/"))
    ):
        raise ValueError("--base-url must be an absolute http(s) URL without credentials, query, or fragment")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/", "", ""))


def prepare_output(requested: Path) -> Path:
    """Refuse source paths and unowned nonempty destinations; do not clear files."""
    if requested.is_symlink():
        raise ValueError("--output must not be a symbolic link")
    output = requested.resolve()
    protected = [
        ROOT / name for name in (
            "site", "scripts", "docs", "src", "tests", "configs", "notebooks",
            "images", ".git", ".github", ".venv", ".agents", ".codex", ".aws",
        )
    ]
    if output == ROOT or output in ROOT.parents or ".git" in output.parts:
        raise ValueError("--output must not be the repository, its parent, or a Git directory")
    if any(output == path or path in output.parents or output in path.parents for path in protected):
        raise ValueError("--output must be separate from source and environment directories")
    if output.exists():
        if not output.is_dir():
            raise ValueError("--output must be a directory")
        marker = output / MARKER
        if any(output.iterdir()) and (
            marker.is_symlink() or not marker.is_file()
            or marker.read_text(encoding="utf-8") != MARKER_CONTENT
        ):
            raise ValueError("refusing a nonempty output without this builder's ownership marker")
    return output


def cpu_install() -> str:
    """Use the README's actual install block rather than maintaining another copy."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = re.search(r"^## Installation\s*\n(.*?)(?=^## |\Z)", readme, re.M | re.S)
    block = re.search(r"```bash\n(.*?)\n```", section.group(1), re.S) if section else None
    if block is None:
        raise ValueError("README.md must contain a bash block in its Installation section")
    commands = block.group(1)
    for required in ("python3.10 -m venv", "torch==1.13.1+cpu", "deepwave==0.0.9"):
        if required not in commands:
            raise ValueError("README CPU installation no longer matches the documented legacy stack")
    return commands


def metrics() -> dict[str, str]:
    """Read only the common initial state and evaluated epoch-120 model RMSE."""
    result = {}
    initial_values = []
    cases = {
        "original_clean_rmse": "FWIGAN-old-metrics.json",
        "package_clean_rmse": "FWIGAN-new-metrics.json",
        "original_noisy_rmse": "FWIGAN_Noisy-old-metrics.json",
        "package_noisy_rmse": "FWIGAN_Noisy-new-metrics.json",
    }
    for label, filename in cases.items():
        data = json.loads((ROOT / "docs/validation/marmousi-120epoch" / filename).read_text(encoding="utf-8"))
        evaluated = {row["epoch"]: row["evaluation"]["model_rmse"] for row in data["rows"] if "evaluation" in row}
        initial, final = evaluated[0], evaluated[120]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (initial, final)):
            raise ValueError(f"non-finite model RMSE in {filename}")
        initial_values.append(initial)
        result[label] = f"{final:.3f}"
    if len(set(initial_values)) != 1:
        raise ValueError("Marmousi comparisons no longer have a shared initial RMSE")
    result["initial_rmse"] = f"{initial_values[0]:.3f}"
    return result


def software_metadata() -> dict:
    """Read supported CodeMeta fields, with a repository-backed fallback."""
    path = ROOT / "codemeta.json"
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(metadata, dict):
        raise ValueError("codemeta.json must contain an object")
    software = {
        "@type": "SoftwareSourceCode",
        "@id": REPOSITORY + "#software",
        "name": "FWIGAN",
        "description": DEFAULT_DESCRIPTION,
        "codeRepository": REPOSITORY,
        "url": REPOSITORY,
        "license": "https://spdx.org/licenses/MIT",
        "programmingLanguage": "Python",
        "runtimePlatform": "Python 3.10",
        "author": [{"@type": "Person", "name": "Muhammad Izzatullah"}],
    }
    # Keep structured data limited to factual software fields shown on the page.
    # Do not inherit dates, ratings, affiliations, or identifiers from elsewhere.
    for field in ("name", "description", "version", "programmingLanguage", "runtimePlatform", "license"):
        value = metadata.get(field)
        if isinstance(value, str) and value.strip():
            software[field] = value
    if software["name"] != "FWIGAN" or software["license"] != "https://spdx.org/licenses/MIT":
        raise ValueError("CodeMeta name/license disagree with this FWIGAN documentation")
    authors = metadata.get("author")
    if isinstance(authors, dict):
        authors = [authors]
    if isinstance(authors, list) and authors:
        normalized = []
        for author in authors:
            if not isinstance(author, dict):
                raise ValueError("CodeMeta author must be a Person object")
            name = author.get("name") or " ".join(
                author[field] for field in ("givenName", "familyName")
                if isinstance(author.get(field), str) and author[field].strip()
            )
            if not isinstance(name, str) or not name.strip():
                raise ValueError("CodeMeta author needs a name")
            normalized.append({"@type": "Person", "name": name})
        software["author"] = normalized
    return software


def structured_data(software: dict, canonical: str) -> str:
    graph = {
        "@context": "https://schema.org",
        "@graph": [
            software,
            {
                "@type": "WebSite", "@id": canonical + "#website",
                "url": canonical, "name": "FWIGAN", "description": PAGE_DESCRIPTION,
                "inLanguage": "en", "about": {"@id": software["@id"]},
            },
            {
                "@type": "WebPage", "@id": canonical + "#webpage",
                "url": canonical, "name": PAGE_TITLE, "description": PAGE_DESCRIPTION,
                "inLanguage": "en", "isPartOf": {"@id": canonical + "#website"},
                "mainEntity": {"@id": software["@id"]},
            },
        ],
    }
    # Prevent a metadata string from ending the JSON-LD script element.
    return json.dumps(graph, ensure_ascii=False, indent=2).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def build(output: Path, canonical: str) -> None:
    software = software_metadata()
    image_source = ROOT / "docs/assets" / IMAGE_NAME
    with image_source.open("rb") as image:
        header = image.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("the documented Marmousi image must be a PNG")
    width, height = struct.unpack(">II", header[16:24])
    author_names = ", ".join(author["name"] for author in software["author"])
    record = [software["name"]]
    if "version" in software:
        record.append("version " + software["version"])
    record.extend((software["programmingLanguage"], software["runtimePlatform"]))
    values = {
        "page_title": PAGE_TITLE,
        "page_description": PAGE_DESCRIPTION,
        "canonical_url": canonical,
        "image_url": canonical + "assets/" + IMAGE_NAME,
        "image_width": str(width),
        "image_height": str(height),
        "author_names": author_names,
        "software_description": software["description"],
        "software_record": "; ".join(record),
        "cpu_install": cpu_install(),
        **metrics(),
    }
    escaped = {key: html.escape(value, quote=True) for key, value in values.items()}
    escaped["structured_data"] = structured_data(software, canonical)
    page = Template((SOURCE / "index.html").read_text(encoding="utf-8")).substitute(escaped)
    sitemap = ElementTree.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    ElementTree.SubElement(ElementTree.SubElement(sitemap, "url"), "loc").text = canonical
    files = {
        Path("index.html"): page,
        Path("sitemap.xml"): ElementTree.tostring(sitemap, encoding="unicode", xml_declaration=True) + "\n",
        Path(".nojekyll"): "",
        Path(MARKER): MARKER_CONTENT,
    }
    llms_source = SOURCE / "llms.txt"
    if llms_source.is_file():
        files[Path("llms.txt")] = llms_source.read_text(encoding="utf-8")
    destinations = [*files, Path("assets/styles.css"), Path("assets") / IMAGE_NAME]
    for relative in destinations:
        destination = output / relative
        if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents if parent != output and output in parent.parents):
            raise ValueError(f"refusing symbolic link in generated destination: {relative}")
        if destination.exists() and not destination.is_file():
            raise ValueError(f"generated destination is not a file: {relative}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    for relative, content in files.items():
        (output / relative).write_text(content, encoding="utf-8")
    shutil.copyfile(SOURCE / "styles.css", output / "assets/styles.css")
    shutil.copyfile(image_source, output / "assets" / IMAGE_NAME)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site", help="empty or builder-owned output directory (default: repository _site)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="absolute site URL, including a project path when needed")
    args = parser.parse_args()
    try:
        canonical = base_url(args.base_url)
        output = prepare_output(args.output)
        build(output, canonical)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Site build failed: {error}", file=sys.stderr)
        return 1
    print(f"Built FWIGAN documentation in {output}")
    print(f"Canonical URL: {canonical}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
