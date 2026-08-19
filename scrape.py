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


def load_data(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    return parse_simple_yaml(text)


def parse_scalar_value(value: str):
    value = value.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    if value and value[0] in "\"'":
        return json.loads(value)
    return value


def parse_simple_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict | list]] = [(-1, root)]
    pending_key: tuple[dict, str, int] | None = None

    lines = [
        (len(line) - len(line.lstrip(" ")), line.strip())
        for line in text.splitlines()
        if line.strip()
    ]

    index = 0
    while index < len(lines):
        indent, line = lines[index]

        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        if pending_key is not None and indent <= pending_key[2]:
            pending_key = None

        if pending_key is not None:
            holder, key, key_indent = pending_key
            if line.startswith("-"):
                holder[key] = []
                stack.append((key_indent, holder[key]))
            else:
                holder[key] = {}
                stack.append((key_indent, holder[key]))
            pending_key = None
            continue

        parent = stack[-1][1]

        if line == "-":
            if not isinstance(parent, list):
                raise ValueError(f"expected list, got {type(parent).__name__}")
            item: dict = {}
            parent.append(item)
            stack.append((indent, item))
            index += 1
            continue

        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"expected list, got {type(parent).__name__}")
            item = {}
            parent.append(item)
            stack.append((indent, item))
            key, _, rest = line[2:].partition(":")
            if not key.strip():
                raise ValueError(f"invalid list item: {line!r}")
            if rest.strip():
                item[key.strip()] = parse_scalar_value(rest)
            else:
                pending_key = (item, key.strip(), indent)
            index += 1
            continue

        key, _, rest = line.partition(":")
        if not key.strip():
            raise ValueError(f"invalid line: {line!r}")
        if not isinstance(parent, dict):
            raise ValueError(f"expected dict, got {type(parent).__name__}")

        key = key.strip()
        rest = rest.strip()
        if rest:
            parent[key] = parse_scalar_value(rest)
        else:
            pending_key = (parent, key, indent)
        index += 1

    return root


def format_model_id(provider: str, name: str) -> str:
    return f"{provider.lower()}/{name.lower()}"


def format_availability(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def index_models(data: dict) -> dict[tuple[str, str], dict]:
    models: dict[tuple[str, str], dict] = {}
    for provider, provider_models in data.get("providers", {}).items():
        for model in provider_models:
            models[(provider, model["name"])] = model
    return models


def region_index(model: dict) -> dict[str, dict]:
    return {region["region"]: region for region in model.get("regions", [])}


def format_region_list(regions: list[str], max_show: int = 8) -> str:
    if len(regions) <= max_show:
        return ", ".join(regions)
    shown = ", ".join(regions[:max_show])
    return f"{shown}, ... (+{len(regions) - max_show} more)"


def diff_regions(old_model: dict, new_model: dict) -> list[str]:
    old_regions = region_index(old_model)
    new_regions = region_index(new_model)
    lines: list[str] = []

    added = sorted(set(new_regions) - set(old_regions))
    removed = sorted(set(old_regions) - set(new_regions))
    if added:
        lines.append(f"added regions ({len(added)}): {format_region_list(added)}")
    if removed:
        lines.append(f"removed regions ({len(removed)}): {format_region_list(removed)}")

    field_changes: dict[tuple[str, str, str], list[str]] = {}
    for region in sorted(set(old_regions) & set(new_regions)):
        old_entry = old_regions[region]
        new_entry = new_regions[region]
        for field in ("location", "in_region", "geo", "global"):
            old_value = old_entry.get(field)
            new_value = new_entry.get(field)
            if old_value != new_value:
                key = (
                    field,
                    format_availability(old_value),
                    format_availability(new_value),
                )
                field_changes.setdefault(key, []).append(region)

    for (field, old_value, new_value), regions in sorted(field_changes.items()):
        if len(regions) == 1:
            lines.append(f"{regions[0]} {field}: {old_value} -> {new_value}")
        else:
            lines.append(
                f"{field}: {old_value} -> {new_value} in {len(regions)} regions "
                f"({format_region_list(regions)})"
            )

    return lines


def diff_models(old: dict, new: dict) -> tuple[list[str], list[str], list[list[str]]]:
    old_models = index_models(old)
    new_models = index_models(new)
    added: list[str] = []
    removed: list[str] = []
    changed: list[list[str]] = []

    for key in sorted(set(new_models) - set(old_models)):
        provider, name = key
        added.append(f"{format_model_id(provider, name)}: added model")

    for key in sorted(set(old_models) - set(new_models)):
        provider, name = key
        removed.append(f"{format_model_id(provider, name)}: removed model")

    for key in sorted(set(old_models) & set(new_models)):
        provider, name = key
        details = diff_regions(old_models[key], new_models[key])
        if details:
            changed.append([format_model_id(provider, name), *details])

    return added, removed, changed


def format_commit_message(old: dict, new: dict) -> str:
    added, removed, changed = diff_models(old, new)
    headline = (
        f"Changed: {len(changed)}, removed: {len(removed)}, added: {len(added)}"
    )

    body_lines: list[str] = []
    body_lines.extend(added)
    body_lines.extend(removed)
    for entry in changed:
        body_lines.append(entry[0] + ":")
        for detail in entry[1:]:
            body_lines.append(f"  {detail}")

    if body_lines:
        return f"{headline}\n\n" + "\n".join(body_lines)
    return headline


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
    parser.add_argument(
        "--print-commit-message",
        action="store_true",
        help="Print a commit message comparing previous and new output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = args.output or Path(f"bedrock-availability.{args.format}")

    previous = None
    if output_path.exists():
        try:
            previous = load_data(output_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: could not read {output_path}: {exc}", file=sys.stderr)
            return 1

    try:
        data = scrape(args.url)
    except (URLError, TimeoutError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_output(data, output_path, args.format)

    if args.print_commit_message:
        if previous is None:
            model_count = sum(len(models) for models in data["providers"].values())
            print(f"Changed: 0, removed: 0, added: {model_count}\n")
            print(f"Initial import of {model_count} models")
        else:
            print(format_commit_message(previous, data))

    model_count = sum(len(models) for models in data["providers"].values())
    print(
        f"Wrote {model_count} models from {len(data['providers'])} providers "
        f"to {output_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
