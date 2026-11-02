"""Guardrail: finds replies stating prices, stock or order data that no tool or policy backs."""

import re
from decimal import Decimal

from emporio_agent.agent.tools import list_promotions, look_up_order, search_catalog

DATA_TOOL_NAMES = {search_catalog.name, list_promotions.name, look_up_order.name}
MONEY_AMOUNT = re.compile(r"R\$\s*(\d[\d.]*(?:,\d{1,2})?)")
STOCK_PHRASE = re.compile(
    r"em estoque|fora de estoque|sem estoque|esgotad[oa]s?|\d+\s+unidades?", re.IGNORECASE
)
ORDER_STATUS = re.compile(
    r"pedido\s*(?:n[º°o]\.?\s*|#)?\d+[^.!?\n]{0,80}?"
    r"(?:enviad|entregue|confirmad|pendente|cancelad|despachad|a caminho|em trânsito)",
    re.IGNORECASE,
)
TRACKING_CODE = re.compile(r"\bBR[0-9A-Z]{9}BR\b")


def needs_tool_call(reply_text: str, policy_excerpts: str, called_tool_names: set[str]) -> bool:
    """Tell whether a reply states store data while no data tool was called in its turn."""
    if called_tool_names & DATA_TOOL_NAMES:
        return False
    return bool(unsupported_claims(reply_text, policy_excerpts))


def unsupported_claims(reply_text: str, policy_excerpts: str) -> list[str]:
    """List the price, stock, order and tracking claims of a reply that no policy holds."""
    return [
        *money_claims_not_in_policies(reply_text, policy_excerpts),
        *phrases_not_in_policies(STOCK_PHRASE, reply_text, policy_excerpts),
        *phrases_not_in_policies(ORDER_STATUS, reply_text, policy_excerpts),
        *phrases_not_in_policies(TRACKING_CODE, reply_text, policy_excerpts),
    ]


def money_claims_not_in_policies(reply_text: str, policy_excerpts: str) -> list[str]:
    """List the R$ amounts in a reply that do not appear in the policy excerpts."""
    policy_amounts = {as_decimal(amount) for amount in MONEY_AMOUNT.findall(policy_excerpts)}
    reply_amounts = MONEY_AMOUNT.findall(reply_text)
    return [amount for amount in reply_amounts if as_decimal(amount) not in policy_amounts]


def phrases_not_in_policies(
    pattern: re.Pattern, reply_text: str, policy_excerpts: str
) -> list[str]:
    """List the reply's matches of a pattern that the policy excerpts do not contain."""
    lowercase_excerpts = policy_excerpts.casefold()
    matches = [match.group(0) for match in pattern.finditer(reply_text)]
    return [phrase for phrase in matches if phrase.casefold() not in lowercase_excerpts]


def as_decimal(brazilian_amount: str) -> Decimal:
    """Read an amount written the Brazilian way ('1.234,56', '80', '500.') as a Decimal."""
    without_thousands_dots = brazilian_amount.replace(".", "")
    return Decimal(without_thousands_dots.replace(",", "."))
