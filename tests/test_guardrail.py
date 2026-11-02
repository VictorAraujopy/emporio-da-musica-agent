"""Tests for the guardrail: which replies state store data that no tool or policy backs."""

from emporio_agent.agent.guardrail import needs_tool_call

POLICY_EXCERPTS = (
    "[Section 5.1]\n5. Política de Frete\n5.1 Entregas\n"
    "• Frete grátis para pedidos acima de R$ 500,00.\n• Taxa fixa de R$ 35,00.\n"
    "[Section 3.1]\n• Parcelamento em até 3x (exceto abaixo de R$\n50,00).\n"
    "[Section 7.3]\nProduto fora de estoque: sugerir alternativas.\n"
    "[Section 5.3]\nFormato: BR seguido de 9 caracteres e BR (exemplo: BR4K7M2X9P1BR)."
)
NO_TOOLS: set[str] = set()


def test_a_price_without_any_tool_needs_a_tool_call() -> None:
    """A product price stated with no tool call in the turn is not backed by anything."""
    assert needs_tool_call("O violão custa R$ 599,90.", POLICY_EXCERPTS, NO_TOOLS)


def test_a_price_after_a_catalog_search_is_backed() -> None:
    """A data tool called in the turn backs the reply."""
    assert not needs_tool_call("O violão custa R$ 599,90.", POLICY_EXCERPTS, {"search_catalog"})


def test_the_discount_calculator_alone_does_not_back_a_price() -> None:
    """apply_discount computes but does not look the price up, so it backs nothing."""
    assert needs_tool_call("No PIX fica R$ 569,91.", POLICY_EXCERPTS, {"apply_discount"})


def test_amounts_quoted_from_the_policies_need_no_tool() -> None:
    """Amounts in the turn's policy excerpts are backed, even split across lines there."""
    reply = "Frete grátis acima de R$ 500, senão R$ 35,00; e nada abaixo de R$ 50,00 parcela."
    assert not needs_tool_call(reply, POLICY_EXCERPTS, NO_TOOLS)


def test_stock_claims_need_a_tool_unless_quoted_from_the_policies() -> None:
    """'Em estoque' is a data claim; 'fora de estoque' as a policy rule is not."""
    assert needs_tool_call("Tem 12 unidades em estoque.", POLICY_EXCERPTS, NO_TOOLS)
    policy_rule = "Se estiver fora de estoque, sugiro outro."
    assert not needs_tool_call(policy_rule, POLICY_EXCERPTS, NO_TOOLS)


def test_an_order_status_needs_a_tool_but_asking_for_the_email_does_not() -> None:
    """Saying how an order stands is a data claim; asking for its email is not."""
    assert needs_tool_call("Seu pedido 8 foi enviado ontem.", POLICY_EXCERPTS, NO_TOOLS)
    asking_for_email = "Pra ver o pedido 8, me passa o e-mail?"
    assert not needs_tool_call(asking_for_email, POLICY_EXCERPTS, NO_TOOLS)


def test_a_tracking_code_needs_a_tool_unless_it_is_the_policy_example() -> None:
    """A real tracking code comes from an order; the manual's example code does not."""
    assert needs_tool_call("Seu rastreio é BRAB1234567BR.", POLICY_EXCERPTS, NO_TOOLS)
    assert not needs_tool_call("O formato é como BR4K7M2X9P1BR.", POLICY_EXCERPTS, NO_TOOLS)


def test_a_reply_without_store_data_needs_no_tool() -> None:
    """Greetings and out-of-scope answers state nothing to check."""
    assert not needs_tool_call("Oi! Como posso te ajudar?", POLICY_EXCERPTS, NO_TOOLS)
