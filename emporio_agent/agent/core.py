"""The agent core: answers one customer message with the store's tools and the policy manual."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    ModelResponse,
    before_model,
    dynamic_prompt,
    wrap_model_call,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    trim_messages,
)
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langsmith import tracing_context
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError

from emporio_agent.agent.guardrail import needs_tool_call
from emporio_agent.agent.prompt import (
    POLICY_SEARCH_UNAVAILABLE,
    build_system_prompt,
    format_policy_excerpts,
)
from emporio_agent.agent.tools import AGENT_TOOLS
from emporio_agent.config import (
    ALLOW_PROVIDER_FALLBACKS,
    CHAT_TIMEOUT_MS,
    EMBEDDING_TIMEOUT_MS,
    OPENROUTER_BASE_URL,
    PROVIDER_DATA_COLLECTION,
    REQUIRE_PROVIDER_PARAMETERS,
    REQUIRE_ZERO_DATA_RETENTION,
    Settings,
    load_settings,
)
from emporio_agent.openrouter_client import create_openrouter_client
from emporio_agent.policies.embeddings import embed_texts
from emporio_agent.policies.index import PolicyIndex, load_policy_index, search_policies
from emporio_agent.store.database import ensure_database

CUSTOMER_MESSAGES_IN_RETRIEVAL_QUERY = 3
HISTORY_MESSAGE_LIMIT = 40
POLICY_SEARCH_ERRORS = (OpenRouterError, httpx.HTTPError, ValueError)


@dataclass(frozen=True)
class TurnContext:
    """What one turn passes to the agent besides the messages."""

    system_prompt: str
    force_tool_call: bool = False


@dataclass(frozen=True)
class AgentCore:
    """Everything loaded once at startup and used by every reply."""

    settings: Settings
    embedding_client: OpenRouter
    policy_index: PolicyIndex
    agent: CompiledStateGraph


def start_agent_core() -> AgentCore:
    """Load the settings, the store database and the policy index, and build the chat agent."""
    settings = load_settings()
    ensure_database()
    embedding_client = create_openrouter_client(settings, EMBEDDING_TIMEOUT_MS)
    policy_index = load_policy_index(embedding_client, settings)
    agent = build_agent(create_chat_model(settings))
    return AgentCore(settings, embedding_client, policy_index, agent)


def create_chat_model(settings: Settings) -> ChatOpenRouter:
    """Create the chat model, on our own OpenRouter client and the fixed provider rules."""
    return ChatOpenRouter(
        model=settings.chat_model,
        api_key=settings.openrouter_api_key,
        base_url=OPENROUTER_BASE_URL,
        client=create_openrouter_client(settings, CHAT_TIMEOUT_MS),
        openrouter_provider=chat_provider_rules(settings),
    )


def build_agent(chat_model: BaseChatModel) -> CompiledStateGraph:
    """Build the tool-calling agent, with per-turn system prompt and in-memory history."""
    return create_agent(
        model=chat_model,
        tools=AGENT_TOOLS,
        middleware=[keep_recent_history, use_this_turns_system_prompt, force_tool_call_on_retry],
        context_schema=TurnContext,
        checkpointer=InMemorySaver(),
    )


def chat_provider_rules(settings: Settings) -> dict[str, object]:
    """Send chat requests only to the configured providers, with the privacy rules from config."""
    provider_rules: dict[str, object] = {
        "order": list(settings.chat_providers),
        "allow_fallbacks": ALLOW_PROVIDER_FALLBACKS,
        "zdr": REQUIRE_ZERO_DATA_RETENTION,
        "data_collection": PROVIDER_DATA_COLLECTION,
        "require_parameters": REQUIRE_PROVIDER_PARAMETERS,
    }
    if settings.chat_quantizations:
        provider_rules["quantizations"] = list(settings.chat_quantizations)
    return provider_rules


@dynamic_prompt
def use_this_turns_system_prompt(request: ModelRequest) -> str:
    """Give the model the system prompt built for this turn: today's date and policy excerpts."""
    return request.runtime.context.system_prompt


@wrap_model_call
def force_tool_call_on_retry(
    request: ModelRequest, call_model: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """On a guardrail retry, make the turn's first model call use a tool."""
    last_message = request.messages[-1] if request.messages else None
    is_first_call_of_turn = isinstance(last_message, HumanMessage)
    if request.runtime.context.force_tool_call and is_first_call_of_turn:
        request = request.override(tool_choice="required")
    return call_model(request)


@before_model
def keep_recent_history(state: AgentState, runtime: Runtime) -> dict | None:
    """Drop the oldest messages past the limit from the saved conversation."""
    messages = state["messages"]
    if len(messages) <= HISTORY_MESSAGE_LIMIT:
        return None
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_history(messages)]}


def recent_history(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Keep the latest messages up to the limit, starting at a customer message."""
    recent_messages = trim_messages(
        messages,
        max_tokens=HISTORY_MESSAGE_LIMIT,
        token_counter=len,
        strategy="last",
        start_on="human",
    )
    if not recent_messages:
        return current_turn_messages(messages, including_customer_message=True)
    return recent_messages


def current_turn_messages(
    messages: list[AnyMessage], including_customer_message: bool = False
) -> list[AnyMessage]:
    """Return the messages after the last customer message, which opened the current turn."""
    customer_positions = [
        position for position, message in enumerate(messages) if isinstance(message, HumanMessage)
    ]
    if not customer_positions:
        return list(messages)
    turn_start = customer_positions[-1]
    if not including_customer_message:
        turn_start += 1
    return messages[turn_start:]


def reply(core: AgentCore, conversation_id: str, customer_message: str) -> str:
    """Answer one customer message within its conversation and return the reply's text."""
    conversation = {"configurable": {"thread_id": conversation_id}}
    retrieval_query = build_retrieval_query(core, conversation, customer_message)
    policy_excerpts = find_policy_excerpts(core, retrieval_query)
    system_prompt = build_system_prompt(date.today(), policy_excerpts)
    turn = TurnContext(system_prompt)
    answer = run_turn(core, conversation, [HumanMessage(customer_message)], turn)
    turn_messages = current_turn_messages(saved_messages(core, conversation))
    if needs_tool_call(answer.text, policy_excerpts, called_tool_names(turn_messages)):
        remove_turn = [RemoveMessage(id=message.id) for message in turn_messages]
        core.agent.update_state(conversation, {"messages": remove_turn})
        retry = TurnContext(system_prompt, force_tool_call=True)
        answer = run_turn(core, conversation, [], retry)
    return answer.text


def run_turn(
    core: AgentCore, conversation: dict, new_messages: list[AnyMessage], turn: TurnContext
) -> AIMessage:
    """Run the agent on the conversation plus any new messages, never traced to LangSmith."""
    with tracing_context(enabled=False):
        result = core.agent.invoke({"messages": new_messages}, config=conversation, context=turn)
    return result["messages"][-1]


def saved_messages(core: AgentCore, conversation: dict) -> list[AnyMessage]:
    """Return the messages saved so far in a conversation."""
    return core.agent.get_state(conversation).values.get("messages", [])


def called_tool_names(turn_messages: list[AnyMessage]) -> set[str]:
    """Return the names of the tools the model called in a turn."""
    return {
        tool_call["name"]
        for message in turn_messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    }


def build_retrieval_query(core: AgentCore, conversation: dict, customer_message: str) -> str:
    """Join the customer's latest messages, so a follow-up keeps the earlier question's topic."""
    earlier_messages = saved_messages(core, conversation)
    earlier_customer_texts = [
        message.text for message in earlier_messages if isinstance(message, HumanMessage)
    ]
    recent_earlier_texts = earlier_customer_texts[-(CUSTOMER_MESSAGES_IN_RETRIEVAL_QUERY - 1) :]
    return "\n".join([*recent_earlier_texts, customer_message])


def find_policy_excerpts(core: AgentCore, retrieval_query: str) -> str:
    """Search the manual for the query; if the search fails, tell the model it is unavailable."""
    try:
        query_vectors = embed_texts(core.embedding_client, [retrieval_query], core.settings)
    except POLICY_SEARCH_ERRORS:
        return POLICY_SEARCH_UNAVAILABLE
    retrieved_chunks = search_policies(core.policy_index, query_vectors[0])
    return format_policy_excerpts(retrieved_chunks)
