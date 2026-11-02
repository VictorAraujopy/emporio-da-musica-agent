"""Tool-calling gate: does the chat model call the right tool, with the right arguments? Real API.

Run by hand from the project root: python -m evals.tool_calling_gate
Each case is a new conversation answered by the real agent; the gate passes only if all pass.
"""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emporio_agent.agent.core import AgentCore

PYDANTIC_V1_WARNING = "Core Pydantic V1 functionality isn't compatible with Python 3.14"


@dataclass(frozen=True)
class GateCase:
    """A customer message, the tool it must trigger (None: no tool) and that tool's arguments."""

    message: str
    expected_tool: str | None
    expected_arguments: dict[str, object] = field(default_factory=dict)


GATE_CASES = (
    GateCase("Quanto custa o Yamaha C40?", "search_catalog", {"name": "c40"}),
    GateCase("Quais produtos estão em promoção?", "list_promotions"),
    GateCase("Tem violão até R$ 700?", "search_catalog", {"max_price_brl": 700}),
    GateCase(
        "Quero ver o pedido 8, meu e-mail é anacarol.ferreira@coldmail.com",
        "look_up_order",
        {"order_id": 8, "email": "anacarol.ferreira@coldmail.com"},
    ),
    GateCase("Qual o status do pedido 8?", None),
    GateCase(
        "Quanto fica R$ 1.000,00 com 5% de desconto?",
        "apply_discount",
        {"price_brl": 1000, "discount_percent": 5},
    ),
    GateCase(
        "Posso parcelar R$ 400 em 5x?",
        "split_into_installments",
        {"total_brl": 400, "installments": 5},
    ),
    GateCase("Vocês aceitam boleto?", None),
    GateCase("Me conta uma piada", None),
)


def main() -> None:
    """Run every gate case on the real agent and print each result and the verdict."""
    warnings.filterwarnings("ignore", message=PYDANTIC_V1_WARNING, category=UserWarning)
    from emporio_agent.agent.core import reply, start_agent_core

    core = start_agent_core()
    passed = 0
    for number, case in enumerate(GATE_CASES, start=1):
        conversation_id = f"gate-{number}"
        reply(core, conversation_id, case.message)
        tool_calls = tool_calls_in_conversation(core, conversation_id)
        case_passed = case_passes(case, tool_calls)
        passed += case_passed
        called = ", ".join(f"{call['name']}({call['args']})" for call in tool_calls) or "no tool"
        print(f"{'PASS' if case_passed else 'FAIL'} | {case.message} -> {called}")
    verdict = "PASSED" if passed == len(GATE_CASES) else "FAILED"
    print(f"\nGATE {verdict}: {passed}/{len(GATE_CASES)}")


def tool_calls_in_conversation(core: "AgentCore", conversation_id: str) -> list[dict]:
    """Return every tool call the model made in a conversation, in order."""
    from langchain_core.messages import AIMessage

    conversation = {"configurable": {"thread_id": conversation_id}}
    messages = core.agent.get_state(conversation).values["messages"]
    return [call for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]


def case_passes(case: GateCase, tool_calls: list[dict]) -> bool:
    """No tool when none is expected; otherwise some call to the tool with matching arguments."""
    if case.expected_tool is None:
        return not tool_calls
    matching_calls = [call for call in tool_calls if call["name"] == case.expected_tool]
    return any(arguments_match(case.expected_arguments, call["args"]) for call in matching_calls)


def arguments_match(expected: dict[str, object], actual: dict[str, object]) -> bool:
    """Numbers must be equal; texts must contain the expected text, ignoring case."""
    for name, expected_value in expected.items():
        actual_value = actual.get(name)
        if isinstance(expected_value, str):
            if expected_value.casefold() not in str(actual_value).casefold():
                return False
        elif actual_value is None or float(actual_value) != float(expected_value):
            return False
    return True


if __name__ == "__main__":
    main()
