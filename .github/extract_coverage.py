import json
import xml.etree.ElementTree as ET
from pathlib import Path


def get_coverage_color(pct):
    if pct >= 90:
        return "brightgreen"
    if pct >= 80:
        return "green"
    if pct >= 70:
        return "yellow"
    if pct >= 60:
        return "orange"
    return "red"


repo_dir = Path(__file__).parent.parent

# Check if coverage.xml exists
coverage_file = repo_dir / "coverage.xml"
if not coverage_file.exists():
    raise FileNotFoundError(f"{str(coverage_file)} not found")

# Parse coverage.xml
tree = ET.parse(coverage_file)
root = tree.getroot()
coverage = float(root.attrib["line-rate"]) * 100

# Create badges directory
badges_dir = repo_dir / "badges"
badges_dir.mkdir(exist_ok=True)

# Generate badge JSON
badge_data = {
    "schemaVersion": 1,
    "label": "coverage",
    "message": f"{coverage:.1f}%",
    "color": get_coverage_color(coverage),
}

with open(badges_dir / "coverage.json", "w", encoding="utf-8") as f:
    json.dump(badge_data, f, indent=2)

print(f"Coverage: {coverage:.1f}%")
