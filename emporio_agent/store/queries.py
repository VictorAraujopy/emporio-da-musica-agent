"""Read-only queries on the store's database, returning typed records."""

import json
import sqlite3
import unicodedata
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path

from emporio_agent.store.database import DATABASE_FILE, open_read_only
from emporio_agent.store.models import (
    ONE_CENT,
    ActivePromotion,
    Order,
    OrderItem,
    OrderStatus,
    Product,
    ProductStatus,
)

PRODUCTS_SQL = """
    SELECT products.product_id, products.name, categories.name AS category,
           products.description, products.price_cents, products.stock_quantity,
           products.status, products.specs,
           promotions.description AS promotion_description,
           promotions.discount_percent AS promotion_discount_percent
    FROM products
    JOIN categories ON categories.category_id = products.category_id
    LEFT JOIN promotions
        ON promotions.product_id = products.product_id AND promotions.is_active = 1
"""

SEARCH_PRODUCTS_SQL = PRODUCTS_SQL + """
    WHERE (:name IS NULL
           OR instr(normalize_text(products.name), normalize_text(:name)) > 0)
      AND (:category IS NULL
           OR normalize_text(categories.name) = normalize_text(:category))
"""

PRODUCTS_ON_PROMOTION_SQL = PRODUCTS_SQL + """
    WHERE promotions.promotion_id IS NOT NULL
"""

ORDER_SQL = """
    SELECT orders.order_id, orders.order_date, orders.status, orders.total_cents,
           orders.payment_method, orders.tracking_code, orders.estimated_delivery,
           orders.notes
    FROM orders
    JOIN customers ON customers.customer_id = orders.customer_id
    WHERE orders.order_id = :order_id AND lower(customers.email) = :email
"""

ORDER_ITEMS_SQL = """
    SELECT products.name AS product_name, order_items.quantity
    FROM order_items
    JOIN products ON products.product_id = order_items.product_id
    WHERE order_items.order_id = :order_id
    ORDER BY products.name
"""

CATEGORY_NAMES_SQL = "SELECT name FROM categories ORDER BY name"


def search_products(
    name: str | None = None,
    category: str | None = None,
    max_price_brl: Decimal | None = None,
    database_file: Path = DATABASE_FILE,
) -> list[Product]:
    """Return the products matching every filter given, in any status, cheapest first."""
    parameters = {"name": name, "category": category}
    rows = fetch_rows(SEARCH_PRODUCTS_SQL, parameters, database_file)
    products = [product_from_row(row) for row in rows]
    if max_price_brl is not None:
        products = [product for product in products if product.current_price_brl <= max_price_brl]
    return cheapest_first(products)


def find_products_on_promotion(database_file: Path = DATABASE_FILE) -> list[Product]:
    """Return the products that have an active promotion, cheapest first."""
    rows = fetch_rows(PRODUCTS_ON_PROMOTION_SQL, {}, database_file)
    products = [product_from_row(row) for row in rows]
    return cheapest_first(products)


def find_order(order_id: int, email: str, database_file: Path = DATABASE_FILE) -> Order | None:
    """Return the order if the email matches its customer's record; otherwise None."""
    normalized_email = email.strip().lower()
    order_parameters = {"order_id": order_id, "email": normalized_email}
    order_rows = fetch_rows(ORDER_SQL, order_parameters, database_file)
    if not order_rows:
        return None
    item_rows = fetch_rows(ORDER_ITEMS_SQL, {"order_id": order_id}, database_file)
    return order_from_rows(order_rows[0], item_rows)


def list_category_names(database_file: Path = DATABASE_FILE) -> list[str]:
    """Return the names of the store's product categories, alphabetically."""
    rows = fetch_rows(CATEGORY_NAMES_SQL, {}, database_file)
    return [row["name"] for row in rows]


def fetch_rows(
    sql: str, parameters: Mapping[str, object], database_file: Path
) -> list[sqlite3.Row]:
    """Run one query on a read-only connection and return all its rows."""
    with open_read_only(database_file) as connection:
        connection.create_function("normalize_text", 1, normalize_text, deterministic=True)
        return connection.execute(sql, parameters).fetchall()


def normalize_text(text: str | None) -> str | None:
    """Return the text in lowercase and without accents, so "Elétrico" matches "eletrico"."""
    if text is None:
        return None
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_accents.casefold()


def cheapest_first(products: list[Product]) -> list[Product]:
    """Return the products sorted by the price the customer pays now, lowest first."""
    return sorted(products, key=lambda product: product.current_price_brl)


def product_from_row(row: sqlite3.Row) -> Product:
    """Build a Product from a row of the products query."""
    return Product(
        product_id=row["product_id"],
        name=row["name"],
        category=row["category"],
        description=row["description"],
        price_brl=cents_to_brl(row["price_cents"]),
        stock_quantity=row["stock_quantity"],
        status=ProductStatus(row["status"]),
        specs=json.loads(row["specs"]),
        promotion=promotion_from_row(row),
    )


def promotion_from_row(row: sqlite3.Row) -> ActivePromotion | None:
    """Build a product's active promotion from its row; None if it has none."""
    if row["promotion_description"] is None:
        return None
    return ActivePromotion(
        description=row["promotion_description"],
        discount_percent=row["promotion_discount_percent"],
    )


def order_from_rows(order_row: sqlite3.Row, item_rows: list[sqlite3.Row]) -> Order:
    """Build an Order from its row and the rows of its items."""
    return Order(
        order_id=order_row["order_id"],
        order_date=date.fromisoformat(order_row["order_date"]),
        status=OrderStatus(order_row["status"]),
        total_brl=cents_to_brl(order_row["total_cents"]),
        payment_method=order_row["payment_method"],
        tracking_code=order_row["tracking_code"],
        estimated_delivery=optional_date(order_row["estimated_delivery"]),
        notes=order_row["notes"],
        items=tuple(order_item_from_row(row) for row in item_rows),
    )


def order_item_from_row(row: sqlite3.Row) -> OrderItem:
    """Build an OrderItem from a row of the order items query."""
    return OrderItem(product_name=row["product_name"], quantity=row["quantity"])


def optional_date(text: str | None) -> date | None:
    """Convert an ISO date written as text to a date; None stays None."""
    if text is None:
        return None
    return date.fromisoformat(text)


def cents_to_brl(cents: int) -> Decimal:
    """Convert whole cents to an amount in reais with two decimal places."""
    return (Decimal(cents) / 100).quantize(ONE_CENT)
