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
