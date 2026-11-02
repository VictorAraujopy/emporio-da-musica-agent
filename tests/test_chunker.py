"""Tests for the policy manual chunking: on a manual written by the test, and on the real PDF."""

import pytest

from emporio_agent.config import PROJECT_ROOT
from emporio_agent.policies.chunker import Chunk, split_manual_into_chunks
from emporio_agent.policies.extract import extract_manual_text

MANUAL_FILE = PROJECT_ROOT / "data" / "store_policies.pdf"
TEST_MANUAL_TEXT = "\n".join(
    [
        "CAPA DO MANUAL",
        "Versão 1.0",
        "1. Sobre a loja",
        "Texto próprio da seção 1.",
        "1.1 Missão",
        "Texto da missão.",
        "1.2 Dados",
        "Texto dos dados.",
        "2. Horário",
        "Texto do horário.",
        "3. Pagamento",
        "3.1 Parcelamento",
        "Texto do parcelamento.",
        "4. Atendimento",
        "4.1 Fluxo",
        "1. Saudação: cumprimentar o cliente.",
        "2. Entendimento: identificar a necessidade.",
        "5. Garantia",
        "Texto da garantia.",
    ]
)


@pytest.fixture(scope="module")
def test_manual_chunks() -> list[Chunk]:
    """Split the manual written by the test once."""
    return split_manual_into_chunks(TEST_MANUAL_TEXT)


def chunk_with_id(chunks: list[Chunk], section_id: str) -> Chunk:
    """Return the only chunk with the given section id."""
    matching_chunks = [chunk for chunk in chunks if chunk.section_id == section_id]
    assert len(matching_chunks) == 1
    return matching_chunks[0]


def test_one_chunk_per_subsection_or_per_section_without_any(
    test_manual_chunks: list[Chunk],
) -> None:
    """Parents with subsections get no chunk of their own; list items are not headings."""
    section_ids = [chunk.section_id for chunk in test_manual_chunks]
    assert section_ids == ["1.1", "1.2", "2", "3.1", "4.1", "5"]


def test_every_subsection_chunk_carries_its_parent_heading_and_text(
    test_manual_chunks: list[Chunk],
) -> None:
    """Each subsection's chunk opens with its parent section's heading and own text."""
    first_chunk = chunk_with_id(test_manual_chunks, "1.1")
    second_chunk = chunk_with_id(test_manual_chunks, "1.2")
    parent_text = "1. Sobre a loja\nTexto próprio da seção 1."
    assert first_chunk.text == f"{parent_text}\n1.1 Missão\nTexto da missão."
    assert second_chunk.text == f"{parent_text}\n1.2 Dados\nTexto dos dados."


def test_a_parent_without_text_of_its_own_adds_only_its_heading(
    test_manual_chunks: list[Chunk],
) -> None:
    """A parent section with no text before its first subsection contributes just its heading."""
    chunk = chunk_with_id(test_manual_chunks, "3.1")
    assert chunk.text == "3. Pagamento\n3.1 Parcelamento\nTexto do parcelamento."


def test_a_section_without_subsections_is_one_chunk(test_manual_chunks: list[Chunk]) -> None:
    """A section with no subsections becomes a chunk with its heading and its text."""
    chunk = chunk_with_id(test_manual_chunks, "2")
    assert chunk.text == "2. Horário\nTexto do horário."


def test_a_numbered_list_stays_in_the_text_of_its_subsection(
    test_manual_chunks: list[Chunk],
) -> None:
    """List items shaped like '1. Saudação:' stay inside the subsection they belong to."""
    chunk = chunk_with_id(test_manual_chunks, "4.1")
    expected_text = (
        "4. Atendimento\n4.1 Fluxo\n"
        "1. Saudação: cumprimentar o cliente.\n2. Entendimento: identificar a necessidade."
    )
    assert chunk.text == expected_text


def test_the_last_section_runs_to_the_end_of_the_text(test_manual_chunks: list[Chunk]) -> None:
    """The last section's text goes to the end of the manual."""
    chunk = chunk_with_id(test_manual_chunks, "5")
    assert chunk.text == "5. Garantia\nTexto da garantia."


def test_text_before_the_first_heading_is_dropped(test_manual_chunks: list[Chunk]) -> None:
    """The cover, before the first section heading, is in no chunk."""
    all_chunks_text = "\n".join(chunk.text for chunk in test_manual_chunks)
    assert "CAPA DO MANUAL" not in all_chunks_text
    assert "Versão 1.0" not in all_chunks_text


def test_subsection_headings_out_of_order_or_from_another_section_are_text() -> None:
    """'1.2' before '1.1', and '2.1' inside section 1, are not headings: they stay as text."""
    manual_text = "\n".join(
        [
            "1. Seção um",
            "1.2 Fora de ordem",
            "2.1 De outra seção",
            "1.1 Primeira",
            "1.2 Segunda",
        ]
    )
    chunks = split_manual_into_chunks(manual_text)
    parent_text = "1. Seção um\n1.2 Fora de ordem\n2.1 De outra seção"
    chunk_texts = [chunk.text for chunk in chunks]
    assert chunk_texts == [f"{parent_text}\n1.1 Primeira", f"{parent_text}\n1.2 Segunda"]


def test_a_section_number_that_jumps_ahead_is_text() -> None:
    """'3.' inside section 1, before any '2.', is not a heading: it stays as text."""
    manual_lines = ["1. Primeira", "3. Item adiantado", "2. Segunda", "Texto da segunda."]
    manual_text = "\n".join(manual_lines)
    chunks = split_manual_into_chunks(manual_text)
    chunk_texts = [chunk.text for chunk in chunks]
    assert chunk_texts == ["1. Primeira\n3. Item adiantado", "2. Segunda\nTexto da segunda."]


def test_headings_with_two_digit_numbers_and_punctuation_are_found() -> None:
    """A section numbered 10 and a title with parentheses, digits and a hyphen are headings."""
    sections_one_to_nine = [f"{number}. Seção {number}" for number in range(1, 10)]
    manual_lines = [
        *sections_one_to_nine,
        "10. Disposições Finais",
        "10.1 Direito de Arrependimento (Compras Online) - 7 dias",
        "Texto da 10.1.",
    ]
    chunks = split_manual_into_chunks("\n".join(manual_lines))
    expected_ids = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10.1"]
    assert [chunk.section_id for chunk in chunks] == expected_ids


def test_text_without_any_section_heading_raises() -> None:
    """Text with no numbered heading cannot be chunked, so it is an error."""
    with pytest.raises(ValueError):
        split_manual_into_chunks("Texto sem nenhuma seção numerada.\nOutra linha.")


def test_the_real_manual_gives_uniquely_numbered_chunks_opening_with_their_section() -> None:
    """Each real chunk has its own number and starts with its section's heading."""
    chunks = split_manual_into_chunks(extract_manual_text(MANUAL_FILE))
    section_ids = [chunk.section_id for chunk in chunks]
    assert chunks != []
    assert len(section_ids) == len(set(section_ids))
    for chunk in chunks:
        section_of_chunk = chunk.section_id.split(".")[0]
        assert chunk.text.startswith(f"{section_of_chunk}. ")


def test_no_line_after_the_cover_is_lost_in_the_real_manual() -> None:
    """Every line from the first section heading on ends up in some chunk."""
    manual_lines = extract_manual_text(MANUAL_FILE).splitlines()
    chunks = split_manual_into_chunks("\n".join(manual_lines))
    first_heading = chunks[0].text.splitlines()[0]
    lines_after_cover = manual_lines[manual_lines.index(first_heading) :]
    chunked_lines = {line for chunk in chunks for line in chunk.text.splitlines()}
    lost_lines = [line for line in lines_after_cover if line not in chunked_lines]
    assert lost_lines == []
