#!/usr/bin/env python3
"""Render a plain-text copy of the complete MLP H1 README."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
content = (ROOT / "README.md").read_text()


def markdown_table_to_text(lines):
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    if len(rows) > 1 and all(set(cell) <= set(":-") and "-" in cell for cell in rows[1]):
        rows.pop(1)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    output = []
    for index, row in enumerate(rows):
        output.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if index == 0:
            output.append("-+-".join("-" * width for width in widths))
    return output


plain = []
lines = content.splitlines()
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("|") and line.endswith("|"):
        block = []
        while i < len(lines) and lines[i].startswith("|") and lines[i].endswith("|"):
            block.append(lines[i])
            i += 1
        plain.extend(markdown_table_to_text(block))
        continue
    heading = re.match(r"^(#{1,6})\s+(.+)$", line)
    if heading:
        title = heading.group(2)
        plain.extend((title, "=" * len(title) if len(heading.group(1)) == 1 else "-" * len(title)))
    elif line.startswith("```"):
        pass
    else:
        line = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"\1 (\2)", line)
        line = line.replace("**", "").replace("`", "")
        plain.append(line)
    i += 1

(ROOT / "README.txt").write_text("\n".join(plain).rstrip() + "\n")
print("README.txt atualizado a partir de README.md")
