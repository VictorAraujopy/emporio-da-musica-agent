"""Tests for the agent core with a scripted chat model: guardrail retry, history and retrieval."""

import dataclasses
import json
import time
from types import SimpleNamespace

import langsmith.utils

import httpx
import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from pydantic import Field

from emporio_agent.agent import tools as tools_module
from emporio_agent.agent.core import (
    HISTORY_MESSAGE_LIMIT,
    AgentCore,
    build_agent,
    build_retrieval_query,
    create_chat_model,
    current_turn_messages,
    find_policy_excerpts,
    recent_history,
    reply,
)
from emporio_agent.agent.prompt import POLICY_SEARCH_UNAVAILABLE
from emporio_agent.config import Settings
from emporio_agent.policies.chunker import Chunk
from emporio_agent.policies.index import PolicyIndex

TEST_SETTINGS = Settings(
    openrouter_api_key="test-key",
    chat_model="test/chat-model",
    chat_providers=("test-chat-provider",),
    chat_quantizations=(),
    embedding_model="test/embedding-model",
    embedding_providers=("test-embedding-provider",),
)
POLICY_TEXT = "5. Frete\n5.1 Entregas\nFrete grátis para pedidos acima de R$ 500,00."
CONVERSATION = {"configurable": {"thread_id": "test-conversation"}}


class ScriptedChatModel(FakeMessagesListChatModel):
    """A fake chat model answering from a script and recording each call's settings."""

    tool_choices: list[object] = Field(default_factory=list)
    tracing_states: list[bool] = Field(default_factory=list)

    def _generate(self, *args: object, **kwargs: object) -> object:
        """Record whether LangSmith tracing is on in this call, then answer from the script."""
        self.tracing_states.append(langsmith.utils.tracing_is_enabled())
        return super()._generate(*args, **kwargs)

    def bind_tools(
        self, tools: object, *, tool_choice: object = None, **kwargs: object
    ) -> object:
        """Record the tool_choice of this model call and keep answering from the script."""
        self.tool_choices.append(tool_choice)
        return self


class StubEmbeddingsEndpoint:
    """Stands in for client.embeddings: every text gets the vector (1, 0)."""

    def generate(self, input: list[str], **request_options: object) -> SimpleNamespace:
        """Return one (1, 0) vector per text, shaped like the real response."""
        data = [SimpleNamespace(embedding=[1.0, 0.0], index=i) for i in range(len(input))]
        return SimpleNamespace(data=data)


class FailingEmbeddingsEndpoint:
    """Stands in for client.embeddings when the network is down."""

    def generate(self, input: list[str], **request_options: object) -> SimpleNamespace:
        """Fail like a dropped connection."""
        raise httpx.ConnectError("network down")


def core_with(chat_model: ScriptedChatModel, embeddings: object) -> AgentCore:
    """Build a real agent core around a scripted chat model and a one-chunk policy index."""
    policy_index = PolicyIndex(
        chunks=(Chunk("5.1", POLICY_TEXT),), vectors=np.array([[1.0, 0.0]], dtype=np.float32)
    )
    embedding_client = SimpleNamespace(embeddings=embeddings)
    return AgentCore(TEST_SETTINGS, embedding_client, policy_index, build_agent(chat_model))


def test_a_price_stated_without_a_tool_is_retried_with_a_forced_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unbacked reply is dropped, and the retry's first model call must use a tool."""
    monkeypatch.setattr(tools_module.queries, "search_products", lambda **filters: [])
    catalog_call = {"name": "search_catalog", "args": {"name": "violão"}, "id": "call-1"}
    chat_model = ScriptedChatModel(
        responses=[
            AIMessage("O violão custa R$ 599,90."),
            AIMessage("", tool_calls=[catalog_call]),
            AIMessage("Consultei o catálogo: esse violão não está cadastrado."),
        ]
    )
    core = core_with(chat_model, StubEmbeddingsEndpoint())
    answer = reply(core, "test-conversation", "Quanto custa o violão?")
    saved_messages = core.agent.get_state(CONVERSATION).values["messages"]
    saved_texts = [message.text for message in saved_messages]
    assert answer == "Consultei o catálogo: esse violão não está cadastrado."
    assert chat_model.tool_choices[1] == "required"
    assert chat_model.tool_choices.count("required") == 1
    assert "O violão custa R$ 599,90." not in saved_texts


def test_a_reply_backed_by_the_policies_is_not_retried() -> None:
    """An amount quoted from the turn's policy excerpts needs no tool and no retry."""
    chat_model = ScriptedChatModel(responses=[AIMessage("Frete grátis acima de R$ 500,00!")])
    core = core_with(chat_model, StubEmbeddingsEndpoint())
    answer = reply(core, "test-conversation", "O frete é grátis?")
    assert answer == "Frete grátis acima de R$ 500,00!"
    assert "required" not in chat_model.tool_choices


def test_a_failed_policy_search_tells_the_model_instead_of_ending_the_turn() -> None:
    """When the question cannot be embedded, the model is told the search is unavailable."""
    core = core_with(ScriptedChatModel(responses=[]), FailingEmbeddingsEndpoint())
    assert find_policy_excerpts(core, "Qual o prazo de troca?") == POLICY_SEARCH_UNAVAILABLE


def test_the_retrieval_query_carries_the_previous_customer_message() -> None:
    """A follow-up like 'e no cartão?' is searched together with the question before it."""
    chat_model = ScriptedChatModel(responses=[AIMessage("No PIX tem desconto, sim!")])
    core = core_with(chat_model, StubEmbeddingsEndpoint())
    reply(core, "test-conversation", "Tem desconto no PIX?")
    query = build_retrieval_query(core, CONVERSATION, "E no cartão?")
    assert query == "Tem desconto no PIX?\nE no cartão?"


def long_conversation_with_tool_calls(turns: int) -> list[AnyMessage]:
    """Make a conversation where each turn has a question, a tool call, its result, a reply."""
    messages: list[AnyMessage] = []
    for turn in range(turns):
        call = {"name": "search_catalog", "args": {}, "id": f"call-{turn}"}
        messages.append(HumanMessage(f"pergunta {turn}"))
        messages.append(AIMessage("", tool_calls=[call]))
        messages.append(ToolMessage("resultado", tool_call_id=f"call-{turn}"))
        messages.append(AIMessage(f"resposta {turn}"))
    return messages


def test_the_history_cap_never_splits_a_tool_call_from_its_result() -> None:
    """The kept history fits the limit, starts at a customer message and has no orphan result."""
    short_last_turn = [HumanMessage("pergunta final"), AIMessage("resposta final")]
    conversation = long_conversation_with_tool_calls(turns=15) + short_last_turn
    kept = recent_history(conversation)
    kept_calls = [call for m in kept if isinstance(m, AIMessage) for call in m.tool_calls]
    kept_call_ids = {call["id"] for call in kept_calls}
    result_ids = {m.tool_call_id for m in kept if isinstance(m, ToolMessage)}
    assert len(kept) <= HISTORY_MESSAGE_LIMIT
    assert isinstance(kept[0], HumanMessage)
    assert result_ids <= kept_call_ids
    assert kept[-1].text == "resposta final"


def test_a_discount_on_a_price_never_looked_up_reruns_the_turn_with_a_forced_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A price no tool returned in the conversation, only discounted, is erased and redone."""
    monkeypatch.setattr(tools_module.queries, "search_products", lambda **filters: [])
    discount_call = {
        "name": "apply_discount",
        "args": {"price_brl": 599.9, "discount_percent": 5},
        "id": "call-1",
    }
    catalog_call = {"name": "search_catalog", "args": {"name": "C40"}, "id": "call-2"}
    chat_model = ScriptedChatModel(
        responses=[
            AIMessage("", tool_calls=[discount_call]),
            AIMessage("No PIX fica R$ 569,91."),
            AIMessage("", tool_calls=[catalog_call]),
            AIMessage("Consultei o catálogo antes de responder."),
        ]
    )
    core = core_with(chat_model, StubEmbeddingsEndpoint())
    answer = reply(core, "test-conversation", "E no PIX?")
    saved = core.agent.get_state(CONVERSATION).values["messages"]
    saved_calls = [
        call["name"] for message in saved if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    assert answer == "Consultei o catálogo antes de responder."
    assert chat_model.tool_choices[2] == "required"
    assert saved_calls == ["search_catalog"]


def test_the_current_turn_starts_after_the_last_customer_message() -> None:
    """The turn is found by the last customer message, so a trimmed history does not shift it."""
    call = {"name": "search_catalog", "args": {}, "id": "call-1"}
    messages = [
        HumanMessage("primeira"),
        AIMessage("resposta"),
        HumanMessage("segunda"),
        AIMessage("", tool_calls=[call]),
        ToolMessage("resultado", tool_call_id="call-1"),
        AIMessage("resposta final"),
    ]
    turn_texts = [message.text for message in current_turn_messages(messages)]
    assert turn_texts == ["", "resultado", "resposta final"]


def test_a_single_turn_longer_than_the_cap_keeps_its_customer_message() -> None:
    """When no customer message fits in the cap, the whole current turn is kept."""
    tool_rounds: list[AnyMessage] = []
    for round_number in range(HISTORY_MESSAGE_LIMIT):
        call = {"name": "search_catalog", "args": {}, "id": f"call-{round_number}"}
        tool_rounds.append(AIMessage("", tool_calls=[call]))
        tool_rounds.append(ToolMessage("resultado", tool_call_id=f"call-{round_number}"))
    conversation = [HumanMessage("antiga"), AIMessage("ok"), HumanMessage("atual"), *tool_rounds]
    kept = recent_history(conversation)
    assert kept[0].text == "atual"
    assert len(kept) == len(tool_rounds) + 1


def test_conversations_are_never_traced_to_langsmith(monkeypatch: pytest.MonkeyPatch) -> None:
    """LANGSMITH_TRACING set in the shell does not turn tracing on during a turn."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://127.0.0.1:9")
    langsmith.utils.get_env_var.cache_clear()
    chat_model = ScriptedChatModel(responses=[AIMessage("Oi! Como posso ajudar?")])
    core = core_with(chat_model, StubEmbeddingsEndpoint())
    reply(core, "test-conversation", "Oi")
    langsmith.utils.get_env_var.cache_clear()
    assert chat_model.tracing_states == [False]


def test_a_price_looked_up_in_an_earlier_turn_can_be_reused_without_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Data a tool returned earlier in the conversation backs the reply: no forced retry."""
    monkeypatch.setattr(tools_module.queries, "search_products", lambda **filters: [])
    catalog_call = {"name": "search_catalog", "args": {"name": "C40"}, "id": "call-1"}
    discount_call = {
        "name": "apply_discount",
        "args": {"price_brl": 599.9, "discount_percent": 5},
        "id": "call-2",
    }
    chat_model = ScriptedChatModel(
        responses=[
            AIMessage("", tool_calls=[catalog_call]),
            AIMessage("O C40 custa R$ 599,90."),
            AIMessage("", tool_calls=[discount_call]),
            AIMessage("No PIX fica R$ 569,91."),
        ]
    )
    core = core_with(chat_model, StubEmbeddingsEndpoint())
    reply(core, "test-conversation", "Quanto custa o C40?")
    answer = reply(core, "test-conversation", "E no PIX?")
    assert answer == "No PIX fica R$ 569,91."
    assert "required" not in chat_model.tool_choices


def test_the_guardrail_retry_is_not_traced_to_langsmith_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Erasing the unbacked turn happens with tracing off too, even with LANGSMITH_TRACING set."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://127.0.0.1:9")
    langsmith.utils.get_env_var.cache_clear()
    monkeypatch.setattr(tools_module.queries, "search_products", lambda **filters: [])
    catalog_call = {"name": "search_catalog", "args": {"name": "violão"}, "id": "call-1"}
    chat_model = ScriptedChatModel(
        responses=[
            AIMessage("O violão custa R$ 599,90."),
            AIMessage("", tool_calls=[catalog_call]),
            AIMessage("Consultei o catálogo."),
        ]
    )
    core = core_with(chat_model, StubEmbeddingsEndpoint())
    tracing_during_erase = []
    original_update_state = core.agent.update_state

    def record_tracing_then_update(*args: object, **kwargs: object) -> object:
        """Note whether tracing is on while the turn is erased, then erase it."""
        tracing_during_erase.append(langsmith.utils.tracing_is_enabled())
        return original_update_state(*args, **kwargs)

    monkeypatch.setattr(core.agent, "update_state", record_tracing_then_update)
    reply(core, "test-conversation", "Quanto custa o violão?")
    langsmith.utils.get_env_var.cache_clear()
    assert tracing_during_erase == [False]
    assert chat_model.tracing_states == [False, False, False]


def test_an_order_status_after_only_a_catalog_search_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A product lookup earlier does not back an order status: the turn is redone."""
    monkeypatch.setattr(tools_module.queries, "search_products", lambda **filters: [])
    catalog_call = {"name": "search_catalog", "args": {"name": "C40"}, "id": "call-1"}
    chat_model = ScriptedChatModel(
        responses=[
            AIMessage("", tool_calls=[catalog_call]),
            AIMessage("O C40 custa R$ 599,90."),
            AIMessage("Seu pedido 1234 foi enviado."),
            AIMessage("Pra ver o pedido, preciso do e-mail cadastrado."),
        ]
    )
    core = core_with(chat_model, StubEmbeddingsEndpoint())
    reply(core, "test-conversation", "Quanto custa o C40?")
    answer = reply(core, "test-conversation", "E o meu pedido 1234?")
    assert answer == "Pra ver o pedido, preciso do e-mail cadastrado."
    assert chat_model.tool_choices[-1] == "required"

CHAT_SETTINGS = dataclasses.replace(TEST_SETTINGS, chat_quantizations=("fp8",))


def chat_completion(text: str) -> httpx.Response:
    """Build the answer the chat endpoint gives with this text."""
    message = {"role": "assistant", "content": text}
    choice = {"index": 0, "message": message, "finish_reason": "stop"}
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    body = {"id": "test", "object": "chat.completion", "created": 0, "model": "test"}
    body["system_fingerprint"] = "test"
    return httpx.Response(200, json={**body, "choices": [choice], "usage": usage})


def test_chat_requests_carry_the_privacy_rules_and_no_session_from_the_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chat request pins providers with the fixed rules; the shell's session is ignored."""
    monkeypatch.setenv("OPENROUTER_SESSION_ID", "shell-session")
    sent_bodies = []

    def record_and_answer(request: httpx.Request) -> httpx.Response:
        """Keep the request body and answer with a short reply."""
        sent_bodies.append(json.loads(request.content))
        return chat_completion("Oi!")

    chat_model = create_chat_model(CHAT_SETTINGS)
    chat_model.client.sdk_configuration.client = httpx.Client(
        transport=httpx.MockTransport(record_and_answer)
    )
    chat_model.invoke("Oi")
    expected_provider_rules = {
        "order": ["test-chat-provider"],
        "allow_fallbacks": False,
        "zdr": True,
        "data_collection": "deny",
        "require_parameters": True,
        "quantizations": ["fp8"],
    }
    assert sent_bodies[0]["provider"] == expected_provider_rules
    assert "session_id" not in sent_bodies[0]


def test_a_chat_request_is_sent_once_with_a_30_second_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 503 asking for an hour's wait is not obeyed: one attempt, then the error."""
    attempts = []
    waits = []

    def unavailable_for_an_hour(request: httpx.Request) -> httpx.Response:
        """Answer the first request with 503 and a one-hour Retry-After; any retry succeeds."""
        attempts.append(request)
        if len(attempts) > 1:
            return chat_completion("Oi!")
        error_body = {"error": {"code": 503, "message": "unavailable"}}
        return httpx.Response(503, headers={"Retry-After": "3600"}, json=error_body)

    monkeypatch.setattr(time, "sleep", waits.append)
    chat_model = create_chat_model(CHAT_SETTINGS)
    chat_model.client.sdk_configuration.client = httpx.Client(
        transport=httpx.MockTransport(unavailable_for_an_hour)
    )
    with pytest.raises(Exception):
        chat_model.invoke("Oi")
    assert len(attempts) == 1
    assert waits == []
    assert chat_model.client.sdk_configuration.timeout_ms == 30_000
