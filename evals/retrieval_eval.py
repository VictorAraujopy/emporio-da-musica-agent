"""Retrieval eval: does the policy search find the right manual section? Calls the real API.

Run by hand from the project root: python -m evals.retrieval_eval
"""

from dataclasses import dataclass

import numpy as np
from openrouter import OpenRouter

from emporio_agent.config import EMBEDDING_TIMEOUT_MS, Settings, load_settings
from emporio_agent.openrouter_client import create_openrouter_client
from emporio_agent.policies.embeddings import embed_texts
from emporio_agent.policies.index import POLICY_MIN_SIMILARITY, PolicyIndex, load_policy_index


@dataclass(frozen=True)
class PolicyQuestion:
    """A customer question and the manual sections that answer it (all of them, if required)."""

    text: str
    expected_sections: tuple[str, ...]
    needs_every_section: bool = False


POLICY_QUESTIONS = (
    PolicyQuestion("Qual o endereço da loja?", ("1.2",)),
    PolicyQuestion("Qual o CNPJ da empresa?", ("1.2",)),
    PolicyQuestion("Vocês abrem no domingo?", ("2",)),
    PolicyQuestion("Até que horas vocês ficam abertos no sábado?", ("2",)),
    PolicyQuestion("Vocês funcionam em feriado?", ("2",)),
    PolicyQuestion("Aceitam boleto?", ("3",)),
    PolicyQuestion("Tem desconto pagando à vista?", ("3",)),
    PolicyQuestion("Quanto tempo demora pra compensar o boleto?", ("3",)),
    PolicyQuestion("Consigo dividir em 10 vezes no cartão?", ("3.1", "3")),
    PolicyQuestion("Se eu parcelar em 5x, qual o valor mínimo de cada parcela?", ("3.1",)),
    PolicyQuestion("Posso pagar uma parte no pix e o resto no cartão?", ("3.1",)),
    PolicyQuestion("Dá pra parcelar uma compra de R$ 120 em 3x?", ("3.1",)),
    PolicyQuestion("Me arrependi da compra, posso devolver meu pedido?", ("4.1",)),
    PolicyQuestion("Comprei pelo site e não curti, tenho quantos dias pra desistir?", ("4.1",)),
    PolicyQuestion("Em quanto tempo recebo o reembolso se devolver?", ("4.1",)),
    PolicyQuestion("Quem paga o frete se eu devolver por arrependimento?", ("4.1",)),
    PolicyQuestion("Minha guitarra chegou com defeito, o que eu faço?", ("4.2",)),
    PolicyQuestion(
        "Passou um mês e meu teclado parou de funcionar, ainda troco na loja?", ("4.2", "8.1")
    ),
    PolicyQuestion("Queria trocar meu violão por um de outra cor, dá?", ("4.3",)),
    PolicyQuestion("Posso trocar a boquilha do meu saxofone?", ("4.4",)),
    PolicyQuestion("Fiz uma regulagem especial na guitarra, ainda posso trocar?", ("4.4",)),
    PolicyQuestion("A entrega é de graça aqui em Campo Grande?", ("5.1",)),
    PolicyQuestion("Quanto custa o frete pra um pedido de R$ 300 aqui na cidade?", ("5.1",)),
    PolicyQuestion("Moro em Cuiabá, quanto tempo leva pra chegar?", ("5.2",)),
    PolicyQuestion("Vocês mandam pelos Correios?", ("5.2",)),
    PolicyQuestion("Vocês entregam bateria acústica pra outro estado?", ("5.2",)),
    PolicyQuestion("O instrumento chegou todo amassado da transportadora, e agora?", ("5.2",)),
    PolicyQuestion("Como eu acompanho onde está minha encomenda?", ("5.3",)),
    PolicyQuestion("Qual o formato do código de rastreio?", ("5.3",)),
    PolicyQuestion("Quando vai ter Black Friday?", ("6.1",)),
    PolicyQuestion("Vai ter promoção no Dia do Músico?", ("6.1",)),
    PolicyQuestion("Posso juntar dois descontos na mesma compra?", ("6.2",)),
    PolicyQuestion(
        "Pagando no pix um produto que já está em promoção, ganho os 5% também?",
        ("6.2", "3"),
        needs_every_section=True,
    ),
    PolicyQuestion("Vi uma promoção mês passado, ainda tá valendo?", ("7.3", "6.2")),
    PolicyQuestion("Vocês vendem cordas e palhetas?", ("1", "7.1")),
    PolicyQuestion("Qual o número do WhatsApp de vocês?", ("1.2", "7")),
    PolicyQuestion("Fiz uma reclamação, em quanto tempo vocês respondem?", ("7.3",)),
    PolicyQuestion("Qual o prazo de garantia dos instrumentos?", ("8.1", "8.2")),
    PolicyQuestion("Se o violão cair e quebrar, a garantia cobre?", ("8.3", "4.2")),
    PolicyQuestion("A garantia cobre desgaste dos trastes?", ("8.3",)),
    PolicyQuestion("O que vocês fazem com os meus dados pessoais?", ("9",)),
    PolicyQuestion("Quero que apaguem meus dados do cadastro", ("9",)),
)
OFF_TOPIC_QUESTIONS = (
    "Quem ganhou o jogo do Flamengo ontem?",
    "Me passa uma receita de bolo de cenoura",
    "Qual é a capital da Austrália?",
    "Me ajuda a resolver uma equação de segundo grau",
    "Vai chover amanhã em São Paulo?",
    "Escreve um poema sobre o mar",
    "Qual o preço do dólar hoje?",
    "Me conta uma piada",
    "Quem é o melhor guitarrista de todos os tempos?",
    "Qual acorde eu devo aprender primeiro no violão?",
    "Me indica uma música fácil pra tocar no teclado",
    "Como afinar um violão de ouvido?",
    "Qual a diferença entre jazz e blues?",
    "Quem compôs a Nona Sinfonia?",
    "Qual o melhor app pra aprender bateria?",
    "Que banda de rock você recomenda?",
)
UNCOVERED_QUESTIONS = (
    "Vocês dão aula de violão?",
    "Vocês compram instrumento usado?",
    "Vocês fazem conserto de guitarra?",
    "Tem estacionamento na loja?",
)


def main() -> None:
    """Search the real policy index with every question and print the retrieval metrics."""
    settings = load_settings()
    client = create_openrouter_client(settings, EMBEDDING_TIMEOUT_MS)
    index = load_policy_index(client, settings)
    policy_scores = similarities(index, client, settings, [q.text for q in POLICY_QUESTIONS])
    off_topic_best = similarities(index, client, settings, list(OFF_TOPIC_QUESTIONS)).max(axis=1)
    uncovered_best = similarities(index, client, settings, list(UNCOVERED_QUESTIONS)).max(axis=1)
    relevant = relevant_scores(index, policy_scores)
    print(f"{len(index.chunks)} chunks | {len(POLICY_QUESTIONS)} policy questions")
    for top_k in (1, 3, 5):
        print(f"right section in the top {top_k}: {count_hits(index, policy_scores, top_k)}")
    print(f"lowest relevant score: {min(relevant):.3f}")
    print(f"highest off-topic score: {off_topic_best.max():.3f}")
    accuracy = cutoff_accuracy(relevant, list(off_topic_best), POLICY_MIN_SIMILARITY)
    print(f"cutoff {POLICY_MIN_SIMILARITY:.2f} separates policy from off-topic: {accuracy:.0%}")
    print(f"uncovered questions, best scores: {[round(float(s), 3) for s in uncovered_best]}")


def similarities(
    index: PolicyIndex, client: OpenRouter, settings: Settings, questions: list[str]
) -> np.ndarray:
    """Return a matrix with the cosine of each question (row) against each chunk (column)."""
    question_vectors = embed_texts(client, questions, settings)
    return question_vectors @ index.vectors.T


def answers(section_id: str, expected_section: str) -> bool:
    """A chunk answers an expected section when it is that section or one of its subsections."""
    return section_id == expected_section or section_id.startswith(expected_section + ".")


def count_hits(index: PolicyIndex, policy_scores: np.ndarray, top_k: int) -> str:
    """Count the questions whose expected sections are among the top_k chunks."""
    hits = 0
    for question, scores in zip(POLICY_QUESTIONS, policy_scores):
        top_sections = [index.chunks[i].section_id for i in np.argsort(scores)[::-1][:top_k]]
        found = [
            any(answers(section, expected) for section in top_sections)
            for expected in question.expected_sections
        ]
        hits += all(found) if question.needs_every_section else any(found)
    return f"{hits}/{len(POLICY_QUESTIONS)}"


def relevant_scores(index: PolicyIndex, policy_scores: np.ndarray) -> list[float]:
    """Return, per question, the score of its best answering chunk (worst one if all needed)."""
    relevant = []
    for question, scores in zip(POLICY_QUESTIONS, policy_scores):
        per_section = [
            best_score_for_section(index, scores, expected)
            for expected in question.expected_sections
        ]
        relevant.append(min(per_section) if question.needs_every_section else max(per_section))
    return relevant


def best_score_for_section(
    index: PolicyIndex, scores: np.ndarray, expected_section: str
) -> float:
    """Return the best score among the chunks that answer an expected section."""
    return max(
        float(scores[i])
        for i, chunk in enumerate(index.chunks)
        if answers(chunk.section_id, expected_section)
    )


def cutoff_accuracy(relevant: list[float], off_topic: list[float], cutoff: float) -> float:
    """Share of questions the cutoff puts on the right side: policy above it, off-topic below."""
    policy_kept = sum(score >= cutoff for score in relevant)
    off_topic_dropped = sum(score < cutoff for score in off_topic)
    return (policy_kept + off_topic_dropped) / (len(relevant) + len(off_topic))


if __name__ == "__main__":
    main()
