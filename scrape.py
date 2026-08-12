#!/usr/bin/env python3
"""Scrape Bedrock model regional availability from AWS documentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from lxml import html

DEFAULT_URL = (
    "https://docs.aws.amazon.com/bedrock/latest/userguide/"
    "models-region-compatibility.html"
)
PROVIDER_HEADING_PREFIX = "model-regions-"
INFERENCE_COLUMNS = ("in_region", "geo", "global")


def fetch_page(url: str, timeout: float = 60.0) -> bytes:
    request = Request(url, headers={"User-Agent": "bedrock-availability-scraper/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_availability(cell) -> bool | str:
    img = cell.find(".//img")
    if img is not None:
        src = img.get("src", "")
        if "icon-yes" in src:
            return True
        if "icon-no" in src:
            return False

    text = cell.text_content().strip()
    if not text:
        return False
    return text


def parse_region(cell) -> dict[str, str]:
    code_el = cell.find(".//code")
    if code_el is None:
        raise ValueError(f"region cell missing code element: {cell.text_content()!r}")

    region = code_el.text_content().strip()
    location = cell.text_content().strip()
    location = location.removeprefix(region).strip()
    if location.startswith("(") and location.endswith(")"):
        location = location[1:-1].strip()

    return {"region": region, "location": location}


def parse_model_table(table) -> dict:
    caption = table.find("caption")
    if caption is None:
        raise ValueError("table missing caption")

    link = caption.find("a")
    if link is not None:
        name = link.text_content().strip()
        model_card = link.get("href", "").removeprefix("./")
    else:
        name = caption.text_content().strip()
        model_card = None

    header_cells = table.xpath(".//thead//th")
    columns = [cell.text_content().strip().lower() for cell in header_cells]
    if columns[0] != "region" or len(columns) != 4:
        raise ValueError(f"unexpected table columns: {columns}")

    regions: list[dict] = []
    for row in table.xpath(".//tr[td]"):
        cells = row.xpath("./td")
        if len(cells) != 4:
            continue

        entry = parse_region(cells[0])
        for column_name, cell in zip(INFERENCE_COLUMNS, cells[1:]):
            entry[column_name] = parse_availability(cell)
        regions.append(entry)

    model = {"name": name, "regions": regions}
    if model_card:
        model["model_card"] = model_card
    return model


def parse_providers(root) -> dict[str, list[dict]]:
    providers: dict[str, list[dict]] = {}

    for heading in root.xpath(f'//h2[starts-with(@id, "{PROVIDER_HEADING_PREFIX}")]'):
        provider = heading.text_content().strip()
        models: list[dict] = []

        node = heading.getnext()
        while node is not None and node.tag != "h2":
            for table in node.xpath(".//table"):
                models.append(parse_model_table(table))
            node = node.getnext()

        providers[provider] = models

    return providers


def scrape(url: str) -> dict:
    document = html.fromstring(fetch_page(url))
    providers = parse_providers(document)
    if not providers:
        raise ValueError("no provider sections found; page structure may have changed")

    return {
        "source_url": url,
        "providers": providers,
    }


def dump_yaml(data: dict, indent: int = 0) -> str:
    lines: list[str] = []

    def render(value, level: int) -> None:
        prefix = "  " * level
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    render(item, level + 1)
                else:
                    lines.append(f"{prefix}{key}: {format_scalar(item)}")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    render(item, level + 1)
                else:
                    lines.append(f"{prefix}- {format_scalar(item)}")
        else:
            lines.append(f"{prefix}{format_scalar(value)}")

    render(data, indent)
    return "\n".join(lines) + "\n"


def format_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)

    text = str(value)
    if (
        not text
        or text[0] in "-?:"
        or any(ch in text for ch in ":#{}[]&*!|>'\"%@`")
        or text in {"true", "false", "null"}
    ):
        return json.dumps(text, ensure_ascii=False)
    return text


def write_output(data: dict, output_path: Path, output_format: str) -> None:
    if output_format == "json":
        content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    else:
        content = dump_yaml(data)

    output_path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape Bedrock model regional availability from AWS docs.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Documentation URL to scrape (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (default: bedrock-availability.{json|yaml})",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("json", "yaml"),
        default="yaml",
        help="Output format (default: yaml)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = args.output or Path(f"bedrock-availability.{args.format}")

    try:
        data = scrape(args.url)
    except (URLError, TimeoutError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_output(data, output_path, args.format)

    model_count = sum(len(models) for models in data["providers"].values())
    print(
        f"Wrote {model_count} models from {len(data['providers'])} providers "
        f"to {output_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
