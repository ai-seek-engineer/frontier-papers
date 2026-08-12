#!/usr/bin/env python3
"""Copy Hugo content to a staging directory and protect TeX from Markdown.

The source Markdown is intentionally left untouched.  Goldmark can interpret
parts of a multiline TeX block as Markdown (for example, a standalone ``=``
as a Setext heading, or ``_`` as emphasis).  This pass wraps recognised math
in raw HTML elements before Hugo sees the document.  MathJax still receives
the original TeX delimiters at runtime.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import sys
from pathlib import Path


FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>`{3,}|~{3,})(?P<info>[^\n\r]*)(?P<newline>\r?\n)?$")
BLOCK_OPEN_RE = re.compile(r"^(?P<indent>\s*)(?P<delimiter>\\\[|\$\$)\s*(?P<newline>\r?\n)?$")
BLOCK_CLOSE_RE = re.compile(r"^\s*(?P<delimiter>\\\]|\$\$)\s*(?:\r?\n)?$")


def escaped_tex(value: str) -> str:
    """Escape TeX as HTML text while keeping it meaningful to MathJax."""

    return html.escape(value, quote=False)


def math_inline(delimiter_left: str, body: str, delimiter_right: str) -> str:
    return (
        '<span class="math-inline">'
        f"{delimiter_left}{escaped_tex(body)}{delimiter_right}"
        "</span>"
    )


def math_block(delimiter_left: str, body: str, delimiter_right: str, newline: str) -> str:
    # Keep the delimiters in a raw HTML block so Goldmark cannot reinterpret
    # operators, underscores, emphasis markers, or angle brackets in the TeX.
    return (
        '<div class="math-block">\n'
        f"{delimiter_left}\n{escaped_tex(body)}"
        f"{delimiter_right}\n</div>{newline}"
    )


def protect_single_line_block(line: str, path: Path, line_number: int, warnings: list[str]) -> str:
    r"""Protect \[...\] or $$...$$ blocks written on one Markdown line."""

    for left, right in (("\\[", "\\]"), ("$$", "$$")):
        if left not in line:
            continue
        start = line.find(left)
        closing = closing_delimiter(line, start + len(left), right)
        if closing < 0:
            # A delimiter-only line is the opening line of a multiline block;
            # let the block parser below handle it.
            if line.strip() == left:
                return line
            warnings.append(f"{path}:{line_number}: unclosed {left} block formula")
            return line
        body = line[start + len(left) : closing]
        newline = "\r\n" if line.endswith("\r\n") else "\n"
        protected = math_block(left, body.strip(), right, newline)
        return line[:start] + protected.rstrip("\r\n") + line[closing + len(right) :]
    return line


def closing_delimiter(text: str, start: int, delimiter: str) -> int:
    """Find an unescaped closing delimiter, ignoring escaped dollar signs."""

    cursor = start
    while True:
        cursor = text.find(delimiter, cursor)
        if cursor < 0:
            return -1
        if delimiter == "$" and cursor > 0 and text[cursor - 1] == "\\":
            cursor += len(delimiter)
            continue
        return cursor


def looks_like_dollar_math(body: str) -> bool:
    """Reject ordinary currency/text pairs while retaining common TeX."""

    if not body or body != body.strip() or "\n" in body:
        return False
    return bool(
        re.search(
            r"(?:\\[A-Za-z]+|[_^{}]|\\frac|\\sum|\\prod|\\min|\\max|[=<>]|\\[()\[\]])",
            body,
        )
    )


def protect_inline(line: str, path: Path, line_number: int, warnings: list[str]) -> str:
    """Protect inline TeX while leaving inline code spans unchanged."""

    output: list[str] = []
    cursor = 0
    while cursor < len(line):
        if line[cursor] == "`":
            end = cursor
            while end < len(line) and line[end] == "`":
                end += 1
            delimiter = line[cursor:end]
            closing = line.find(delimiter, end)
            if closing < 0:
                output.append(line[cursor:])
                break
            output.append(line[cursor : closing + len(delimiter)])
            cursor = closing + len(delimiter)
            continue

        if line.startswith("\\(", cursor):
            closing = closing_delimiter(line, cursor + 2, "\\)")
            if closing < 0:
                warnings.append(f"{path}:{line_number}: unclosed \\( inline formula")
                output.append(line[cursor:])
                break
            body = line[cursor + 2 : closing]
            output.append(math_inline("\\(", body, "\\)"))
            cursor = closing + 2
            continue

        # Support existing $...$ content without treating $$ as inline math.
        if (
            line[cursor] == "$"
            and not line.startswith("$$", cursor)
            and (cursor == 0 or not line[cursor - 1].isalnum())
        ):
            closing = closing_delimiter(line, cursor + 1, "$")
            if closing < 0:
                output.append(line[cursor:])
                break
            body = line[cursor + 1 : closing]
            if looks_like_dollar_math(body) and (
                closing + 1 == len(line) or not line[closing + 1].isalnum()
            ):
                output.append(math_inline("$", body, "$"))
                cursor = closing + 1
                continue

        output.append(line[cursor])
        cursor += 1

    return "".join(output)


def protect_markdown(text: str, path: Path, warnings: list[str]) -> str:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    index = 0
    fence_marker = ""

    while index < len(lines):
        line = lines[index]
        fence = FENCE_RE.match(line)

        if fence_marker:
            output.append(line)
            if fence and fence.group("marker")[0] == fence_marker[0] and len(fence.group("marker")) >= len(fence_marker):
                fence_marker = ""
            index += 1
            continue

        if fence:
            fence_marker = fence.group("marker")
            output.append(line)
            index += 1
            continue

        single_line_block = protect_single_line_block(line, path, index + 1, warnings)
        if single_line_block != line:
            output.append(single_line_block)
            index += 1
            continue

        block_open = BLOCK_OPEN_RE.match(line)
        if block_open:
            left = block_open.group("delimiter")
            right = "\\]" if left == "\\[" else "$$"
            body: list[str] = []
            index += 1
            closed = False

            while index < len(lines):
                candidate = lines[index]
                if BLOCK_CLOSE_RE.match(candidate) and candidate.strip() == right:
                    closed = True
                    index += 1
                    break
                body.append(candidate)
                index += 1

            if not closed:
                warnings.append(f"{path}:{index}: unclosed {left} block formula")
                output.append(line)
                output.extend(body)
                break

            body_text = "".join(body)
            newline = "\n"
            if lines[index - 1].endswith("\r\n"):
                newline = "\r\n"
            output.append(math_block(left, body_text.rstrip("\r\n"), right, newline))
            continue

        output.append(protect_inline(line, path, index + 1, warnings))
        index += 1

    return "".join(output)


def prepare_content(source: Path, destination: Path) -> list[str]:
    if destination.exists():
        raise RuntimeError(f"destination already exists: {destination}")

    shutil.copytree(source, destination)
    warnings: list[str] = []

    for path in destination.rglob("*.md"):
        relative = path.relative_to(destination)
        original = source / relative
        text = original.read_text(encoding="utf-8")
        path.write_text(protect_markdown(text, relative, warnings), encoding="utf-8")

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source Hugo content directory")
    parser.add_argument("destination", type=Path, help="new staging directory")
    args = parser.parse_args()

    if not args.source.is_dir():
        parser.error(f"source directory not found: {args.source}")

    warnings = prepare_content(args.source, args.destination)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
