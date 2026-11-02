"""Policy manual text extraction: every page in order, without page headers and footers."""

import re
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page

PAGE_HEADER = "Empório da Música Manual de Políticas e Procedimentos"
PAGE_FOOTER = re.compile(r"Página \d+")


def extract_manual_text(pdf_file: Path) -> str:
    """Return the manual's text: the body of every page, in order, joined by line breaks."""
    with pdfplumber.open(pdf_file) as pdf:
        pages_body = [extract_page_body(page) for page in pdf.pages]
    return "\n".join(pages_body)


def extract_page_body(page: Page) -> str:
    """Return one page's text without its running header and footer."""
    page_lines = page.extract_text().splitlines()
    body_lines = drop_header_and_footer(page_lines)
    return "\n".join(body_lines)


def drop_header_and_footer(page_lines: list[str]) -> list[str]:
    """Drop the header if it is the page's first line and the footer if it is the last."""
    body_lines = page_lines
    if body_lines and body_lines[0] == PAGE_HEADER:
        body_lines = body_lines[1:]
    if body_lines and PAGE_FOOTER.fullmatch(body_lines[-1]):
        body_lines = body_lines[:-1]
    return body_lines
