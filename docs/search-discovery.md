# Search discovery and publishing

FWIGAN's main subjects are full waveform inversion, adversarial seismic inversion,
PyTorch, DeepWave and reproducible Marmousi experiments. Use those terms where
they describe the software. The README stays short; detailed answers live in
[the FAQ](faq.md), [workflow guide](workflow.md) and [measured results](validation/marmousi-120epoch.md).

## What the repository provides

- A descriptive README title and introduction, with one real FWIGAN simulation image.
- `CITATION.cff` and `codemeta.json` identifying the software, source, author and MIT license.
- Package metadata linking to the source, documentation and issue tracker.
- A static, accessible project page with visible answers, source links, canonical URL,
  description, social previews and truthful `SoftwareSourceCode` structured data.
- A sitemap and a curated `llms.txt` source index. The latter is optional navigation,
  not an established Google/Bing ranking or AI-citation signal.
- A dependency-free site build and checks for the actual deployable artifact.

Code version `0.1.0` comes from package metadata; it is not a claimed formal release
or publication date. There is no invented software DOI, review score, affiliation
or performance claim.

## Publish the project page

The intended address is **https://izzatum.github.io/FWIGAN/**. It is a deployment
target, not a claim that a page is already live.

1. Enable **Settings → Pages → Source: GitHub Actions** for this repository.
2. Merge the documentation change into `main`. The **Search documentation** workflow
   builds the page and sitemap, checks the artifact, and deploys only from `main`.
   Pull requests build and validate without publishing.
3. Confirm that the page, its image, `sitemap.xml` and `llms.txt` return HTTP 200.
4. Set the repository's **About → Website** to the live address after deployment.

Local inspection needs only Python 3.10 or newer:

```bash
python scripts/build_search_site.py
python scripts/check_search_site.py _site
python -m http.server 8000 --directory _site
```

Open `http://localhost:8000/`. To use a different public address, pass the same
`--base-url` to the builder and checker and update the deployment configuration.
Canonical, social-image and sitemap URLs must describe the actual public location.

## Google and Bing ownership and discovery

Repository files cannot verify ownership of `github.com`. Verify the deployed
**URL-prefix property** `https://izzatum.github.io/FWIGAN/` in
[Google Search Console](https://search.google.com/search-console/) and
[Bing Webmaster Tools](https://www.bing.com/webmasters/) using each service's real
verification instructions. Add a verification tag or file only after receiving
the actual owner token; no placeholder token is included here.

Submit `https://izzatum.github.io/FWIGAN/sitemap.xml` in both verified properties.
Use Search Console's URL Inspection to request indexing of the page. These actions
require the owner's accounts; a successful build or public URL is not proof of
submission, indexing or ranking.

A GitHub project page is under `/FWIGAN/`. Crawlers consult the host's
`https://izzatum.github.io/robots.txt`, not `/FWIGAN/robots.txt`. Do not describe a
file in this project directory as control over the host's crawler policy. A missing
host-root robots file normally permits crawling. Check any existing host-wide rules
before adding restrictions. Googlebot and Bingbot need access to the page and assets.

[IndexNow](https://www.indexnow.org/documentation) is an optional notification path
for participating engines after the page is live and its required ownership key
file is hosted. It does not submit to Google or guarantee indexing. No submission
or verification is claimed by this repository change.

## Measure visibility honestly

Track branded searches (`FWIGAN`, `FWIGAN GitHub`) separately from broader searches
(`full waveform inversion PyTorch`, `adversarial seismic inversion`). Search
position varies with location, intent and time. Use the verified tools' impressions,
clicks, indexed URLs and query reports rather than a single scraped search page.
Bing's AI Performance report can show citations and grounding queries where
available; those are not a ranking or authority score.

Google says ordinary SEO applies to AI Overviews and AI Mode, with no required
special AI file or schema. Clear, crawlable, source-backed answers and consistent
software identity support readers and automated systems; neither this work nor
any repository-only edit can guarantee first place or an AI citation. Avoid
keyword stuffing, synthetic stars/backlinks and unverified numerical claims.

Official references:

- [Google: AI features and your website](https://developers.google.com/search/docs/appearance/ai-features)
- [Google: structured data policies](https://developers.google.com/search/docs/appearance/structured-data/sd-policies)
- [Google: sitemap guidance](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap)
- [Google: site ownership verification](https://support.google.com/webmasters/answer/9008080)
- [Bing: AI Performance](https://blogs.bing.com/webmaster/2026/2/Introducing-AI-Performance-in-Bing-Webmaster-Tools-Public-Preview/)
- [GitHub: repository topics](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics)
