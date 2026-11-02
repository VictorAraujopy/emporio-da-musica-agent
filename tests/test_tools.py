"""Tests for the agent's tools, with the store queries replaced by stubs: text, privacy, math."""

import math
import sqlite3
from datetime import date
from decimal import Decimal

import pytest

from emporio_agent.agent import tools as tools_module
from emporio_agent.agent.tools import (
    INVALID_DISCOUNT,
    INVALID_INSTALLMENTS,
    NO_PRODUCT_FOUND,
    ORDER_NOT_FOUND,
    STORE_DATA_UNAVAILABLE,
    apply_discount,
    format_brl,
    look_up_order,
    search_catalog,
    split_into_installments,
)
from emporio_agent.store.models import (
    ActivePromotion,
    Order,
    OrderItem,
    OrderStatus,
    Product,
    ProductStatus,
)


def sample_product(**changed_fields: object) -> Product:
    """Make a product for a test, with any of its fields changed."""
    fields: dict[str, object] = {
        "product_id": 1,
        "name": "Violão Teste",
        "category": "Violões",
        "description": "Para iniciantes.",
        "price_brl": Decimal("599.90"),
        "stock_quantity": 12,
        "status": ProductStatus.ACTIVE,
        "specs": {"cordas": "nylon"},
        "promotion": None,
    }
    fields.update(changed_fields)
    return Product(**fields)


def sample_order() -> Order:
    """Make an order for a test."""
    return Order(
        order_id=8,
        order_date=date(2026, 2, 10),
        status=OrderStatus.SHIPPED,
        total_brl=Decimal("1299.00"),
        payment_method="pix",
        tracking_code="BRAB1234567BR",
        estimated_delivery=date(2026, 2, 28),
        notes=None,
        items=(OrderItem(product_name="Violão Teste", quantity=1),),
    )


def test_the_price_limit_reaches_the_query_as_an_exact_decimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A float limit from the model becomes the Decimal it reads as, with no float error."""
    received_limits = []

    def record_limit(**filters: object) -> list[Product]:
        """Keep the price limit the tool passed."""
        received_limits.append(filters["max_price_brl"])
        return []

    monkeypatch.setattr(tools_module.queries, "search_products", record_limit)
    search_catalog.invoke({"max_price_brl": 700.1})
    assert received_limits == [Decimal("700.1")]


def test_a_product_shows_its_price_promotion_stock_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prices come in Brazilian format, with the promotional price computed in code."""
    promotion = ActivePromotion(description="Queima de Estoque", discount_percent=15)
    on_promotion = sample_product(promotion=promotion)
    monkeypatch.setattr(tools_module.queries, "search_products", lambda **f: [on_promotion])
    text = search_catalog.invoke({"name": "violão"})
    assert "price: R$ 599,90" in text
    assert "promotion: 15% off (Queima de Estoque), now R$ 509,92" in text
    assert "stock: 12" in text
    assert "status: available" in text


@pytest.mark.parametrize(
    ("status", "stock", "expected_availability"),
    [
        (ProductStatus.ACTIVE, 0, "status: out of stock"),
        (ProductStatus.DISCONTINUED, 13, "status: discontinued"),
        (ProductStatus.COMING_SOON, 2, "status: coming soon"),
    ],
)
def test_a_product_that_cannot_be_bought_says_why(
    monkeypatch: pytest.MonkeyPatch, status: ProductStatus, stock: int, expected_availability: str
) -> None:
    """Out of stock, discontinued and coming soon are told apart, whatever the stock says."""
    product = sample_product(status=status, stock_quantity=stock)
    monkeypatch.setattr(tools_module.queries, "search_products", lambda **f: [product])
    assert expected_availability in search_catalog.invoke({"name": "violão"})


def test_an_empty_search_says_that_nothing_matched(monkeypatch: pytest.MonkeyPatch) -> None:
    """No product found gives a clear sentence, not an empty answer."""
    monkeypatch.setattr(tools_module.queries, "search_products", lambda **f: [])
    assert search_catalog.invoke({"name": "theremin"}) == NO_PRODUCT_FOUND


def test_an_order_is_looked_up_with_its_number_and_email(monkeypatch: pytest.MonkeyPatch) -> None:
    """The query gets both values, and the reply carries the order's own fields."""
    received_lookups = []

    def record_lookup(order_id: int, email: str) -> Order:
        """Keep what the tool asked for and return a test order."""
        received_lookups.append((order_id, email))
        return sample_order()

    monkeypatch.setattr(tools_module.queries, "find_order", record_lookup)
    text = look_up_order.invoke({"order_id": 8, "email": "cliente@jmail.com"})
    assert received_lookups == [(8, "cliente@jmail.com")]
    assert "ordered on: 10/02/2026" in text
    assert "status: shipped" in text
    assert "tracking code: BRAB1234567BR" in text
    assert "estimated delivery: 28/02/2026" in text


def test_a_wrong_email_and_an_unknown_order_get_the_same_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The query returns None for both, so nobody learns which order numbers exist."""
    monkeypatch.setattr(tools_module.queries, "find_order", lambda order_id, email: None)
    assert look_up_order.invoke({"order_id": 8, "email": "outro@jmail.com"}) == ORDER_NOT_FOUND


def test_an_order_number_too_big_for_the_database_is_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pasted boleto line as order number gives the same answer as a missing order."""

    def too_big_for_sqlite(order_id: int, email: str) -> Order:
        """Fail the way SQLite fails on a number over 64 bits."""
        raise OverflowError("Python int too large to convert to SQLite INTEGER")

    monkeypatch.setattr(tools_module.queries, "find_order", too_big_for_sqlite)
    answer = look_up_order.invoke({"order_id": 10**46, "email": "cliente@jmail.com"})
    assert answer == ORDER_NOT_FOUND


def test_apply_discount_rounds_half_up_to_the_cent() -> None:
    """R$ 599,90 with 5% off is 569,905, which commercial rounding makes 569,91."""
    answer = apply_discount.invoke({"price_brl": 599.90, "discount_percent": 5})
    assert answer == "R$ 599,90 with 5% off: R$ 569,91"


@pytest.mark.parametrize(
    ("price", "percent"),
    [(599.90, 0), (599.90, 100), (-1.0, 5), (math.inf, 5), (599.90, math.nan), (1e27, 5)],
)
def test_apply_discount_refuses_a_price_or_percent_out_of_range(
    price: float, percent: float
) -> None:
    """Negative or infinite prices and percents outside (0, 100) are not computed."""
    answer = apply_discount.invoke({"price_brl": price, "discount_percent": percent})
    assert answer == INVALID_DISCOUNT


def test_amounts_are_written_with_brazilian_separators() -> None:
    """Dots between thousands, a comma before the cents."""
    assert format_brl(Decimal("1234567.5")) == "R$ 1.234.567,50"


def test_installments_that_divide_evenly_are_all_equal() -> None:
    """R$ 400 in 5x is five installments of R$ 80, the 4x-to-6x minimum."""
    answer = split_into_installments.invoke({"total_brl": 400, "installments": 5})
    assert answer == "R$ 400,00 in 5x: 5x R$ 80,00"


def test_the_first_installment_absorbs_the_leftover_cent() -> None:
    """R$ 100 in 3x cannot be split evenly, and the installments must still add up to R$ 100."""
    answer = split_into_installments.invoke({"total_brl": 100, "installments": 3})
    assert answer == "R$ 100,00 in 3x: 1x R$ 33,34 + 2x R$ 33,33"


@pytest.mark.parametrize(
    ("total", "installments"), [(100, 0), (100, 61), (-5, 2), (1e27, 2), (0.10, 12)]
)
def test_installments_refuse_a_total_or_count_out_of_range(
    total: float, installments: int
) -> None:
    """Zero or too many installments, a negative or absurd total, or under a cent each."""
    answer = split_into_installments.invoke({"total_brl": total, "installments": installments})
    assert answer == INVALID_INSTALLMENTS


def test_installments_round_down_and_the_first_takes_the_leftover_cents() -> None:
    """R$ 0,05 in 2x: each rounds down to 0,02, and the first takes the cent left over."""
    answer = split_into_installments.invoke({"total_brl": 0.05, "installments": 2})
    assert answer == "R$ 0,05 in 2x: 1x R$ 0,03 + 1x R$ 0,02"


def test_the_first_installment_is_never_negative() -> None:
    """Many installments of a small total still add up, with no negative first installment."""
    answer = split_into_installments.invoke({"total_brl": 10, "installments": 60})
    assert answer == "R$ 10,00 in 60x: 1x R$ 0,56 + 59x R$ 0,16"


def test_a_database_error_gives_a_text_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the store's database fails mid-session, the tool says so and the chat goes on."""

    def broken_database(**filters: object) -> list[Product]:
        """Fail like a database file that was removed."""
        raise sqlite3.OperationalError("no such table: products")

    monkeypatch.setattr(tools_module.queries, "search_products", broken_database)
    assert search_catalog.invoke({"name": "violão"}) == STORE_DATA_UNAVAILABLE
