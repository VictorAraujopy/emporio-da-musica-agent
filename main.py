"""Terminal adapter: reads customer messages and prints the agent's replies."""

import sqlite3
import sys
import uuid
import warnings
from collections.abc import Callable

import httpx
from openrouter.errors import OpenRouterError

from emporio_agent.config import SettingsError

PYDANTIC_V1_WARNING = "Core Pydantic V1 functionality isn't compatible with Python 3.14"
STARTUP_ERRORS = (
    SettingsError,
    OpenRouterError,
    httpx.HTTPError,
    OSError,
    ValueError,
    sqlite3.Error,
)
CHAT_ERRORS = (OpenRouterError, httpx.HTTPError, ValueError)
EXIT_WORDS = {"sair", "exit", "quit"}

STARTING_MESSAGE = "Iniciando o atendimento da Empório da Música..."
STARTUP_FAILED_MESSAGE = "Não foi possível iniciar o atendimento:"
WELCOME_MESSAGE = (
    "Olá! Sou o assistente virtual da Empório da Música. Como posso ajudar?"
    " (digite 'sair' para encerrar)\n"
)
FAILED_TURN_MESSAGE = "Desculpe, tive um problema pra responder agora. Pode mandar de novo?"
GOODBYE_MESSAGE = "\nObrigado pelo contato com a Empório da Música. Até mais!"


def main() -> None:
    """Run the terminal chat with the store's agent until the customer leaves."""
    warnings.filterwarnings("ignore", message=PYDANTIC_V1_WARNING, category=UserWarning)
    from emporio_agent.agent.core import reply, start_agent_core

    print(STARTING_MESSAGE)
    try:
        core = start_agent_core()
    except STARTUP_ERRORS as error:
        sys.exit(f"{STARTUP_FAILED_MESSAGE} {error}")
    except KeyboardInterrupt:
        sys.exit(GOODBYE_MESSAGE)
    print(WELCOME_MESSAGE)
    conversation_id = str(uuid.uuid4())
    try:
        while (customer_message := read_customer_message()) is not None:
            agent_reply = reply_or_apology(reply, core, conversation_id, customer_message)
            print(f"\nEmpório da Música: {agent_reply}\n")
    except KeyboardInterrupt:
        pass
    print(GOODBYE_MESSAGE)


def reply_or_apology(
    reply: Callable[..., str], core: object, conversation_id: str, customer_message: str
) -> str:
    """Return the agent's reply, or an apology when the turn fails or the reply is empty."""
    try:
        agent_reply = reply(core, conversation_id, customer_message)
    except CHAT_ERRORS:
        return FAILED_TURN_MESSAGE
    if not agent_reply.strip():
        return FAILED_TURN_MESSAGE
    return agent_reply


def read_customer_message() -> str | None:
    """Read the customer's next non-empty message; None when they leave."""
    while True:
        try:
            typed_message = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if typed_message.casefold() in EXIT_WORDS:
            return None
        if typed_message:
            return typed_message


if __name__ == "__main__":
    main()
