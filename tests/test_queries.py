"""Tests for the store queries, run on a database built from the project's CSV files."""

from dataclasses import fields
from decimal import Decimal
from pathlib import Path

import pytest

from emporio_agent.store.database import DATA_DIR, ensure_database
from emporio_agent.store.models import Order, ProductStatus
from emporio_agent.store.queries import (
    find_order,
    find_products_on_promotion,
    list_category_names,
    search_products,
)

ORDER_1_EMAIL = "pedro.oliveira@jmail.com"
ANOTHER_CUSTOMERS_EMAIL = "lucas.mendes@jmail.com"


@pytest.fixture(scope="module")
def database_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the store database from the CSV files in a temporary directory."""
    database_file = tmp_path_factory.mktemp("storage") / "store.db"
    ensure_database(database_file, DATA_DIR)
    return database_file


def test_find_order_returns_the_order_when_the_email_matches(database_file: Path) -> None:
    """The order comes back with its own data and its items."""
    order = find_order(1, ORDER_1_EMAIL, database_file)
    assert order is not None
    assert order.order_id == 1
    assert order.total_brl == Decimal("11499.00")
    assert [item.product_name for item in order.items] == ["Martin D-28 Dreadnought Natural"]


def test_find_order_ignores_spaces_and_case_in_the_email(database_file: Path) -> None:
    """An email typed with extra spaces or capital letters still matches."""
    order = find_order(1, "  Pedro.Oliveira@JMAIL.com ", database_file)
    assert order is not None


def test_find_order_with_another_customers_email_returns_none(database_file: Path) -> None:
    """A valid email that belongs to another customer does not open the order."""
    assert find_order(1, ANOTHER_CUSTOMERS_EMAIL, database_file) is None


def test_wrong_email_and_nonexistent_order_give_the_same_result(database_file: Path) -> None:
    """Nobody can tell a valid order number from an invalid one."""
    wrong_email_result = find_order(1, ANOTHER_CUSTOMERS_EMAIL, database_file)
    nonexistent_order_result = find_order(999, ORDER_1_EMAIL, database_file)
    assert wrong_email_result == nonexistent_order_result


def test_order_record_has_no_customer_fields() -> None:
    """An order carries no customer data that could reach the model."""
    order_field_names = {field.name for field in fields(Order)}
    customer_field_names = {"customer_id", "name", "email", "phone", "city"}
    assert order_field_names.isdisjoint(customer_field_names)


def test_search_finds_a_product_by_brand_and_model(database_file: Path) -> None:
    """A lowercase brand and model finds the catalog product."""
    products = search_products(name="takamine gd20", database_file=database_file)
    assert [product.name for product in products] == ["Takamine GD20 Dreadnought Natural"]


def test_search_ignores_accents_and_case(database_file: Path) -> None:
    """The same products come back with or without accents, in any letter case."""
    without_accents = search_products(name="eletrico", database_file=database_file)
    with_accents_in_capitals = search_products(name="ELÉTRICO", database_file=database_file)
    assert without_accents
    assert all("Elétrico" in product.name for product in without_accents)
    assert with_accents_in_capitals == without_accents


def test_search_treats_wildcard_characters_as_text(database_file: Path) -> None:
    """Characters that are wildcards in SQL LIKE match only themselves."""
    assert search_products(name="_", database_file=database_file) == []
    assert search_products(name="%", database_file=database_file) == []


def test_search_filters_by_category(database_file: Path) -> None:
    """A category typed without accents returns only that category's products."""
    products = search_products(category="violoes", database_file=database_file)
    assert len(products) == 35
    assert all(product.category == "Violões" for product in products)


def test_search_filters_by_the_price_the_customer_pays(database_file: Path) -> None:
    """A product above the limit at full price appears if its promotional price fits."""
    within_limit = search_products(
        name="Crafter HT-100", max_price_brl=Decimal("2000"), database_file=database_file
    )
    below_promotional_price = search_products(
        name="Crafter HT-100", max_price_brl=Decimal("1900"), database_file=database_file
    )
    assert [product.current_price_brl for product in within_limit] == [Decimal("1967.18")]
    assert below_promotional_price == []


def test_search_returns_products_cheapest_first(database_file: Path) -> None:
    """Results are sorted by the price the customer pays."""
    products = search_products(category="Violões", database_file=database_file)
    prices = [product.current_price_brl for product in products]
    assert prices == sorted(prices)


def test_search_includes_unavailable_products(database_file: Path) -> None:
    """A discontinued product is found and reported as not available."""
    products = search_products(name="Shelby SN-7C", database_file=database_file)
    assert len(products) == 1
    assert products[0].status == ProductStatus.DISCONTINUED
    assert not products[0].is_available


def test_availability_needs_active_status_and_stock(database_file: Path) -> None:
    """Only an active product with stock is available, whatever the other field says."""
    discontinued_in_stock = search_products(name="Ibanez Bass 2X", database_file=database_file)
    coming_soon_in_stock = search_products(name="Kala KA-ABP", database_file=database_file)
    active_out_of_stock = search_products(name="Giannini GF-3D", database_file=database_file)
    active_in_stock = search_products(name="Yamaha C40", database_file=database_file)
    assert not discontinued_in_stock[0].is_available
    assert not coming_soon_in_stock[0].is_available
    assert not active_out_of_stock[0].is_available
    assert active_in_stock[0].is_available


def test_search_treats_sql_in_the_name_as_text(database_file: Path) -> None:
    """SQL typed into a search is matched as text and never executed."""
    injected_name = "'; DROP TABLE products; --"
    assert search_products(name=injected_name, database_file=database_file) == []
    assert len(search_products(database_file=database_file)) == 65


def test_promotions_list_only_products_with_an_active_promotion(database_file: Path) -> None:
    """Each product on promotion carries its discount and its promotional price."""
    products = find_products_on_promotion(database_file)
    prices_by_name = {product.name: product.current_price_brl for product in products}
    assert prices_by_name == {
        "Kalani KAL-700T Tenor Natural": Decimal("323.10"),
        "Ohana CK-20 Concert Natural": Decimal("439.20"),
        "Crafter HT-100 Folk Aço Natural": Decimal("1967.18"),
        "Taylor 110e Dreadnought Natural": Decimal("5519.08"),
    }


def test_list_category_names(database_file: Path) -> None:
    """Every category in the store comes back by name."""
    category_names = list_category_names(database_file)
    assert len(category_names) == 9
    assert "Violões" in category_names
