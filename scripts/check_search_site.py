"""Check the deployable search page, its evidence and project-path URLs offline."""

import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://izzatum.github.io/FWIGAN/"
REPOSITORY = "https://github.com/izzatum/FWIGAN"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta = {}
        self.links = []
        self.images = []
        self.ids = set()
        self.jsonld = []
        self.visible = []
        self.headings = 0
        self._script = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "meta":
            self.meta[attrs.get("name") or attrs.get("property")] = attrs.get("content", "")
        if tag in ("a", "link") and "href" in attrs:
            self.links.append((tag, attrs))
        if tag == "img":
            self.images.append(attrs)
        if tag == "h1":
            self.headings += 1
        if tag == "script":
            self._script = attrs.get("type", "")

    def handle_endtag(self, tag):
        if tag == "script":
            self._script = None

    def handle_data(self, data):
        if self._script == "application/ld+json":
            self.jsonld.append(json.loads(data))
        elif self._script is None:
            self.visible.append(data)


def check(output, base_url):
    output = output.resolve()
    base = urlsplit(base_url)
    assert base.scheme == "https" and base.netloc and base.path.endswith("/")
    assert not base.query and not base.fragment and not base.username
    page = Page()
    page.feed((output / "index.html").read_text())
    canonical = [a["href"] for tag, a in page.links if tag == "link" and a.get("rel") == "canonical"]
    assert canonical == [base_url], "Canonical URL must match the deployment address"
    assert page.headings == 1, "Expected one main heading"
    assert "noindex" not in page.meta.get("robots", "").lower()
    assert "index" in page.meta.get("robots", "").lower()
    assert "nofollow" not in page.meta.get("robots", "").lower()
    assert 40 <= len(page.meta.get("description", "")) <= 300
    assert "width=device-width" in page.meta.get("viewport", "")
    assert page.meta.get("og:url") == base_url

    def local_file(url):
        parsed = urlsplit(urljoin(base_url, url))
        if parsed.netloc != base.netloc:
            return None
        assert parsed.path.startswith(base.path), "Local URL escapes project prefix: " + url
        relative = unquote(parsed.path[len(base.path):]) or "index.html"
        target = (output / relative).resolve()
        assert target.is_relative_to(output), "Local URL escapes site output"
        assert target.is_file(), "Missing deployed file: " + url
        return target

    checked_links = 0
    for tag, attrs in page.links:
        url = attrs["href"]
        if url.startswith("#"):
            assert url[1:] in page.ids, "Broken page anchor: " + url
        else:
            local_file(url)
            source_prefix = REPOSITORY + "/blob/main/"
            if url.startswith(source_prefix):
                target = (ROOT / unquote(urlsplit(url).path.split("/blob/main/", 1)[1])).resolve()
                assert target.is_relative_to(ROOT) and target.is_file(), "Broken repository link: " + url
        checked_links += 1

    assert len(page.images) == 1, "Show only the FWIGAN simulation image"
    image = page.images[0]
    assert len(image.get("alt", "")) >= 20, "Simulation image needs a descriptive alternative"
    image_file = local_file(image["src"])
    source_image = ROOT / "docs/assets/marmousi-120epoch-velocity-models.png"
    assert image_file and hashlib.sha256(image_file.read_bytes()).digest() == hashlib.sha256(source_image.read_bytes()).digest()
    assert local_file(page.meta["og:image"]) == image_file
    assert local_file(page.meta["twitter:image"]) == image_file

    assert len(page.jsonld) == 1, "Expected one connected structured-data graph"
    graph = page.jsonld[0]["@graph"]
    by_type = {node["@type"]: node for node in graph}
    assert {"SoftwareSourceCode", "WebSite", "WebPage"}.issubset(by_type)
    software = by_type["SoftwareSourceCode"]
    code = json.loads((ROOT / "codemeta.json").read_text())
    assert software["codeRepository"] == REPOSITORY
    assert software.get("version", software.get("softwareVersion")) == code["version"]
    visible = " ".join(" ".join(page.visible).split())
    assert code["version"] in visible and "Izzatullah" in visible
    assert "aggregateRating" not in software and "review" not in software
    assert "torch==1.13.1+cpu" in visible and "deepwave==0.0.9" in visible

    for case in ("FWIGAN-new", "FWIGAN-old", "FWIGAN_Noisy-new", "FWIGAN_Noisy-old"):
        path = ROOT / "docs/validation/marmousi-120epoch" / (case + "-metrics.json")
        final = json.loads(path.read_text())["rows"][-1]
        assert final["epoch"] == 120
        assert f"{final['evaluation']['model_rmse']:.3f}" in visible, "Missing/mismatched evidence: " + case

    sitemap = ElementTree.parse(output / "sitemap.xml")
    urls = [node.text for node in sitemap.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    assert urls == [base_url], "Sitemap must contain the actual canonical page"
    assert (output / ".nojekyll").is_file()
    if (output / "llms.txt").is_file():
        assert REPOSITORY in (output / "llms.txt").read_text()
    return {"status": "passed", "canonical": base_url, "links_checked": checked_links, "simulation_images": 1, "evidence_cases_checked": 4}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()
    print(json.dumps(check(args.output, args.base_url)))
