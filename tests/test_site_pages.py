import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

SITE = Path(__file__).parents[1] / "site"


class _References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        for name in ("href", "src"):
            value = attributes.get(name)
            if value:
                self.paths.append(value)


def test_documentation_is_a_native_multipage_site() -> None:
    index = (SITE / "index.html").read_text(encoding="utf-8")
    assert 'href="documentation.html"' in index
    for name in (
        "documentation.html",
        "method.html",
        "derivation.html",
        "running.html",
        "results.html",
        "reproduction.html",
        "review.html",
        "accuracy.html",
        "refinement.html",
    ):
        assert (SITE / name).is_file()

    derivation = (SITE / "derivation.html").read_text(encoding="utf-8")
    assert "U<sup>*</sup>(η) = 4η + j₀" in derivation
    assert "spacing gain" in derivation
    assert "1.47" in derivation


def test_site_pages_have_no_broken_local_links() -> None:
    missing: list[str] = []
    for page in SITE.glob("*.html"):
        parser = _References()
        parser.feed(page.read_text(encoding="utf-8"))
        for reference in parser.paths:
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = page.parent / parsed.path
            if not target.exists():
                missing.append(f"{page.name}: {reference}")
    assert not missing, f"broken site links: {missing}"


def test_site_keeps_peak_speed_without_illustrative_comparisons() -> None:
    index = (SITE / "index.html").read_text(encoding="utf-8")
    app = (SITE / "app.js").read_text(encoding="utf-8")
    faq = (SITE / "faq.html").read_text(encoding="utf-8")
    styles = (SITE / "styles.css").read_text(encoding="utf-8")
    assert 'id="peak-speed"' in index
    assert 'id="speed-time"' in index
    assert "How is peak speed measured?" in faq
    content = f"{index}\n{app}\n{faq}\n{styles}"
    for removed in (
        "speed-comparison", "comparison-scale", "slowReferences",
        "fingernail", "Arctic coast", "Statue speed", "emoji comparison",
    ):
        assert removed not in content


def test_refinement_pilot_is_documentation_not_a_replacement_flow_run() -> None:
    docs = (SITE / "documentation.html").read_text()
    page = (SITE / "refinement.html").read_text()
    assert 'href="refinement.html"' in docs
    assert 'href="data/refinement-pilot.json"' in page
    assert "not a new Navier–Stokes trajectory" in page
    assert "as-yet-unimplemented momentum transport" in page
    report = json.loads((SITE / "data/refinement-pilot.json").read_text())
    assert report["passed"] and len(report["cases"]) == 9
    row = next(r for r in report["cases"] if r["base_n"] == 128 and r["levels"] == 4)
    assert row["finest_effective_n"] == 1024
    assert row["stored_cells"] == 8388608
    assert row["active_cells"] == 7602176
    assert row["coarse_fine_flux_mismatch"] == 0
    assert row["divergence_after_linf"] < 1e-8
