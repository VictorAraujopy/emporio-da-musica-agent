"""Tests for the policy manual extraction: on pages written by the test, and on the real PDF."""

import re
from collections import Counter
from pathlib import Path

import pdfplumber
import pytest

from emporio_agent.config import PROJECT_ROOT
from emporio_agent.policies.extract import (
    PAGE_HEADER,
    drop_header_and_footer,
    extract_manual_text,
)

MANUAL_FILE = PROJECT_ROOT / "data" / "store_policies.pdf"
PAGE_NUMBER_LINE = re.compile(r"Página \d+")


class StubPage:
    """Stands in for a pdfplumber page: returns the text the test gave it."""

    def __init__(self, text: str) -> None:
        """Keep the text this page will return."""
        self.text = text

    def extract_text(self) -> str:
        """Return the page's text, as pdfplumber's page.extract_text() would."""
        return self.text


class StubPdf:
    """Stands in for what pdfplumber.open returns: a PDF with stub pages, used in a with block."""

    def __init__(self, pages_text: list[str]) -> None:
        """Make one stub page for each text."""
        self.pages = [StubPage(text) for text in pages_text]

    def __enter__(self) -> "StubPdf":
        """Return the PDF itself, as pdfplumber does."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Do nothing: a stub PDF has no file to close."""


def extract_from_stub_pages(monkeypatch: pytest.MonkeyPatch, pages_text: list[str]) -> str:
    """Run the real extraction on a PDF whose pages have the given text."""
    monkeypatch.setattr(pdfplumber, "open", lambda pdf_file: StubPdf(pages_text))
    return extract_manual_text(Path("stub.pdf"))


def test_header_and_footer_are_removed_from_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each page loses its header and footer; only the body text is left."""
    pages_text = [
        f"{PAGE_HEADER}\nprimeira página\nPágina 2",
        f"{PAGE_HEADER}\nsegunda página\nPágina 3",
    ]
    manual_text = extract_from_stub_pages(monkeypatch, pages_text)
    assert manual_text == "primeira página\nsegunda página"


def test_a_page_without_header_or_footer_comes_through_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cover page, with no header or footer, is kept as it is."""
    pages_text = ["CAPA\nendereço da loja", f"{PAGE_HEADER}\n1. Primeira seção\nPágina 2"]
    manual_text = extract_from_stub_pages(monkeypatch, pages_text)
    assert manual_text == "CAPA\nendereço da loja\n1. Primeira seção"


def test_a_section_crossing_a_page_break_stays_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """A heading that ends one page is followed by its text from the next page."""
    pages_text = [
        f"{PAGE_HEADER}\n5.1 Seção anterior\n5.2 Seção que cruza\nPágina 4",
        f"{PAGE_HEADER}\ntexto da 5.2\nPágina 5",
    ]
    manual_text = extract_from_stub_pages(monkeypatch, pages_text)
    expected_lines = ["5.1 Seção anterior", "5.2 Seção que cruza", "texto da 5.2"]
    assert manual_text.splitlines() == expected_lines


def test_drop_header_and_footer_removes_them_at_the_page_edges() -> None:
    """The header as the first line and the footer as the last line are dropped."""
    page_lines = [PAGE_HEADER, "3. Formas de Pagamento", "Página 3"]
    assert drop_header_and_footer(page_lines) == ["3. Formas de Pagamento"]


def test_drop_header_and_footer_removes_a_footer_on_a_page_without_header() -> None:
    """The footer is dropped even when the page has no header."""
    assert drop_header_and_footer(["corpo da página", "Página 12"]) == ["corpo da página"]


def test_drop_header_and_footer_removes_a_header_on_a_page_without_footer() -> None:
    """The header is dropped even when the page has no footer."""
    assert drop_header_and_footer([PAGE_HEADER, "corpo da página"]) == ["corpo da página"]


def test_drop_header_and_footer_keeps_a_first_line_that_only_starts_like_the_header() -> None:
    """A first line that begins with the header's text but goes on is content, so it stays."""
    page_lines = [f"{PAGE_HEADER} — edição revisada", "corpo da página"]
    expected_lines = [f"{PAGE_HEADER} — edição revisada", "corpo da página"]
    assert drop_header_and_footer(page_lines) == expected_lines


def test_drop_header_and_footer_keeps_a_last_line_that_only_starts_like_a_footer() -> None:
    """A content line starting with 'Página' is not a footer, even as the page's last line."""
    page_lines = ["4.4 Itens Não Elegíveis para Troca", "Página 3 do contrato tem a regra"]
    expected_lines = ["4.4 Itens Não Elegíveis para Troca", "Página 3 do contrato tem a regra"]
    assert drop_header_and_footer(page_lines) == expected_lines


def test_drop_header_and_footer_keeps_them_inside_the_page() -> None:
    """Header and footer text in the middle of a page is content, so it stays."""
    page_lines = ["primeira linha", PAGE_HEADER, "Página 3", "última linha"]
    expected_lines = ["primeira linha", PAGE_HEADER, "Página 3", "última linha"]
    assert drop_header_and_footer(page_lines) == expected_lines


def test_drop_header_and_footer_on_an_empty_page() -> None:
    """A page with no text gives no lines, without an error."""
    assert drop_header_and_footer([]) == []


def test_the_real_manual_is_read_without_page_headers_or_footers() -> None:
    """The real PDF gives text with no repeated line (a header left behind would repeat)."""
    manual_lines = extract_manual_text(MANUAL_FILE).splitlines()
    repeated_lines = [line for line, count in Counter(manual_lines).items() if count > 1]
    page_number_lines = [line for line in manual_lines if PAGE_NUMBER_LINE.fullmatch(line)]
    assert manual_lines != []
    assert repeated_lines == []
    assert page_number_lines == []
