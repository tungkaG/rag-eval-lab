"""Adapter for the BQ40Z80 Technical Reference Manual text dump (TI SLUUBT5C).

Turns the page-flattened datasheet into ONE `Document` per numbered section
(`Chapter`/`Appendix` headings and dotted headings such as `7.2`), carrying
the printed page range in metadata so chunkers propagate it to every chunk.
Page footers, running headers, blank-page notices, the table of contents,
the Data Flash summary table (19.17), and everything from the revision
history onward are dropped. There is no ground truth: `evaluation_cases()`
is empty, so this adapter only serves interactive querying.

All BQ40Z80-specific file knowledge stays in this module.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_eval_lab.core.models import Document, EvaluationCase

# A footer line closes printed page N: odd pages end "... BQ40Z80 N", even pages start "N BQ40Z80 ...".
_FOOTER = re.compile(r"^(?:SLUUBT5C – .* BQ40Z80 (\d+)|(\d+) BQ40Z80 SLUUBT5C – .*)$")
_NOISE = ("Submit Document Feedback", "Copyright © ", "This page intentionally left blank.")
_CHAPTER = re.compile(r"^(?:Chapter (\d+)|Appendix ([A-Z])|(Preface))$")
_HEADING = re.compile(r"^([0-9A-Z]+)((?:\.\d+)+) (\S.*)$")
# shortcut: pages 1-12 are the title page and table of contents; re-check if the source revision changes.
_LAST_TOC_PAGE = 12


class Bq40z80DatasetAdapter:
    """Reads the BQ40Z80 datasheet text file into section-level `Document`s."""

    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        self._documents: list[Document] | None = None

    def documents(self) -> list[Document]:
        if self._documents is None:
            self._documents = self._parse()
        return self._documents

    def evaluation_cases(self) -> list[EvaluationCase]:
        return []

    def _parse(self) -> list[Document]:
        sections: list[dict] = []
        seen: set[str] = set()
        chapter = None
        awaiting_title = False
        skipping = False

        for page, line in self._page_lines():
            if line == "Revision History":
                break

            chapter_match = _CHAPTER.match(line)
            if chapter_match:
                chapter = next(group for group in chapter_match.groups() if group)
                skipping = False
                awaiting_title = True
                sections.append({"section": chapter, "title": "", "pages": [page], "lines": []})
                continue
            if awaiting_title:
                sections[-1]["title"] = line
                awaiting_title = False
                continue

            heading_match = _HEADING.match(line)
            number = heading_match and heading_match.group(1) + heading_match.group(2)
            # Only new numbers under the current chapter: skips values like "3.3 V" and wrapped "See Section\n7.4.3 ...".
            if heading_match and heading_match.group(1) == chapter and number not in seen:
                seen.add(number)
                title = heading_match.group(3)
                skipping = title == "Data Flash Summary"
                sections.append({"section": number, "title": title, "pages": [page], "lines": []})
                continue

            if sections and not skipping:
                sections[-1]["lines"].append(line)
                sections[-1]["pages"].append(page)

        return [
            Document(
                id=f"bq40z80_{section['section']}",
                text=f"{section['section']} {section['title']}\n" + "\n".join(section["lines"]),
                metadata={
                    "section": section["section"],
                    "title": section["title"],
                    "page_start": min(section["pages"]),
                    "page_end": max(section["pages"]),
                },
            )
            for section in sections
            if section["lines"]
        ]

    def _page_lines(self) -> list[tuple[int, str]]:
        """Content lines tagged with their printed page; the page number is only known at the footer."""
        result: list[tuple[int, str]] = []
        pending: list[str] = []
        for raw_line in self.dataset_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            footer = _FOOTER.match(line)
            if footer:
                page = int(footer.group(1) or footer.group(2))
                if page > _LAST_TOC_PAGE:
                    result.extend((page, pending_line) for pending_line in pending)
                pending = []
            elif line and not line.startswith(_NOISE) and not line.startswith("www.ti.com ") and not line.endswith(" www.ti.com"):
                pending.append(line)
        return result
