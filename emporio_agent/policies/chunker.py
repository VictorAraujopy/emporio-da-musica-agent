"""Policy manual chunking: one chunk per subsection, with its parent section's text."""

import re
from dataclasses import dataclass

SECTION_HEADING = re.compile(r"(\d+)\. .+")
SUBSECTION_HEADING = re.compile(r"(\d+)\.(\d+) .+")


@dataclass(frozen=True)
class Chunk:
    """A piece of the manual to embed: the id of its section ('3.1', or '2') and its text."""

    section_id: str
    text: str


@dataclass(frozen=True)
class Heading:
    """A heading found in the manual text; subsection_number is None for a section's heading."""

    line_index: int
    section_number: int
    subsection_number: int | None


@dataclass(frozen=True)
class ManualPart:
    """A heading and the text under it, up to the next heading."""

    section_number: int
    subsection_number: int | None
    heading: str
    body: str


def split_manual_into_chunks(manual_text: str) -> list[Chunk]:
    """Split the manual's text into chunks: one per subsection, or one per section without any."""
    lines = manual_text.splitlines()
    headings = find_headings(lines)
    if not headings:
        raise ValueError("No numbered section heading ('1. ...') found in the manual text.")
    parts = split_into_parts(lines, headings)
    return build_chunks(parts)


def find_headings(lines: list[str]) -> list[Heading]:
    """Find section and subsection headings, accepting each only as the next number in order."""
    headings = []
    current_section = 0
    current_subsection = 0
    for line_index, line in enumerate(lines):
        if is_next_section_heading(line, current_section):
            current_section += 1
            current_subsection = 0
            headings.append(Heading(line_index, current_section, None))
        elif is_next_subsection_heading(line, current_section, current_subsection):
            current_subsection += 1
            headings.append(Heading(line_index, current_section, current_subsection))
    return headings


def is_next_section_heading(line: str, current_section: int) -> bool:
    """Tell whether a line is '<N>. Title' with N right after the current section."""
    match = SECTION_HEADING.fullmatch(line)
    return match is not None and int(match.group(1)) == current_section + 1


def is_next_subsection_heading(line: str, current_section: int, current_subsection: int) -> bool:
    """Tell whether a line is '<N>.<M> Title' in the current section, with M the next number."""
    match = SUBSECTION_HEADING.fullmatch(line)
    if match is None:
        return False
    section_number = int(match.group(1))
    subsection_number = int(match.group(2))
    return section_number == current_section and subsection_number == current_subsection + 1


def split_into_parts(lines: list[str], headings: list[Heading]) -> list[ManualPart]:
    """Cut the lines at each heading; the text before the first heading (the cover) is dropped."""
    body_ends = [heading.line_index for heading in headings[1:]] + [len(lines)]
    parts = []
    for heading, body_end in zip(headings, body_ends):
        body_lines = lines[heading.line_index + 1 : body_end]
        part = ManualPart(
            section_number=heading.section_number,
            subsection_number=heading.subsection_number,
            heading=lines[heading.line_index],
            body="\n".join(body_lines),
        )
        parts.append(part)
    return parts


def build_chunks(parts: list[ManualPart]) -> list[Chunk]:
    """Make one chunk per subsection, with its parent's text, or one per section without any."""
    sections = [part for part in parts if part.subsection_number is None]
    chunks = []
    for section in sections:
        subsections = subsections_of(section, parts)
        if subsections:
            chunks.extend(subsection_chunk(section, subsection) for subsection in subsections)
        else:
            chunks.append(Chunk(str(section.section_number), part_text(section)))
    return chunks


def subsections_of(section: ManualPart, parts: list[ManualPart]) -> list[ManualPart]:
    """Return the subsections that belong to a section, in order."""
    return [
        part
        for part in parts
        if part.section_number == section.section_number and part.subsection_number is not None
    ]


def subsection_chunk(section: ManualPart, subsection: ManualPart) -> Chunk:
    """Make a subsection's chunk: its parent section's heading and text, then its own."""
    section_id = f"{subsection.section_number}.{subsection.subsection_number}"
    text = f"{part_text(section)}\n{part_text(subsection)}"
    return Chunk(section_id, text)


def part_text(part: ManualPart) -> str:
    """Return a part's heading, followed by its body when it has one."""
    if not part.body:
        return part.heading
    return f"{part.heading}\n{part.body}"
