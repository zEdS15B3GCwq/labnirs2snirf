"""Generate index.md from readme and template."""

from pathlib import Path
import re


def _inject_readme_into_index() -> None:
    """Generate index.md by injecting readme.md into index.template."""
    project_root = Path(__file__).parents[1]
    readme_path = project_root / "README.md"
    template_path = Path(__file__).parent / "index.template"
    if not (readme_path.exists() and template_path.exists()):
        return

    readme_text = readme_path.read_text(encoding="utf8")
    body = re.search(
        r"(?<=<!-- INDEX_START -->\n).*(?=\n<!-- INDEX_END -->)",
        readme_text,
        flags=re.S,
    ).group(0)

    template_text = template_path.read_text(encoding="utf8")
    index_text = re.sub(
        r"<!-- README_START -->.*?<!-- README_END -->",
        body,
        template_text,
        flags=re.S,
    )

    index_path = Path(__file__).parent / "index.md"
    index_path.write_text(index_text, encoding="utf8")


# perform injection at import/build time
if __name__ == "__main__":
    _inject_readme_into_index()
