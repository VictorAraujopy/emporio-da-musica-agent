"""Initial data load: inserts the rows of the CSV files in data/ into a new database."""

import csv
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path


def seed_database(connection: sqlite3.Connection, data_dir: Path) -> None:
    """Insert every CSV file's rows, each parent table before the tables that reference it."""
    seed_categories(connection, data_dir / "categories.csv")
    seed_products(connection, data_dir / "products.csv")
    seed_promotions(connection, data_dir / "promotions.csv")
    seed_customers(connection, data_dir / "customers.csv")
    seed_orders(connection, data_dir / "orders.csv")
    seed_order_items(connection, data_dir / "order_items.csv")


def seed_categories(connection: sqlite3.Connection, csv_file: Path) -> None:
    """Insert the categories from their CSV file."""
    categories = read_csv_rows(csv_file)
    connection.executemany(
        "INSERT INTO categories (category_id, name, description)"
        " VALUES (:category_id, :name, :description)",
        categories,
    )


def seed_products(connection: sqlite3.Connection, csv_file: Path) -> None:
    """Insert the products from their CSV file."""
    products = [product_for_insert(csv_row) for csv_row in read_csv_rows(csv_file)]
    connection.executemany(
        "INSERT INTO products (product_id, name, category_id, description, price_cents,"
        " stock_quantity, status, specs, created_at)"
        " VALUES (:product_id, :name, :category_id, :description, :price_cents,"
        " :stock_quantity, :status, :specs, :created_at)",
        products,
    )


def seed_promotions(connection: sqlite3.Connection, csv_file: Path) -> None:
    """Insert the promotions from their CSV file."""
    promotions = read_csv_rows(csv_file)
    connection.executemany(
        "INSERT INTO promotions (promotion_id, product_id, discount_percent, description,"
        " is_active)"
        " VALUES (:promotion_id, :product_id, :discount_percent, :description, :is_active)",
        promotions,
    )


def seed_customers(connection: sqlite3.Connection, csv_file: Path) -> None:
    """Insert the customers from their CSV file."""
    customers = read_csv_rows(csv_file)
    connection.executemany(
        "INSERT INTO customers (customer_id, name, phone, email, city)"
        " VALUES (:customer_id, :name, :phone, :email, :city)",
        customers,
    )


def seed_orders(connection: sqlite3.Connection, csv_file: Path) -> None:
    """Insert the orders from their CSV file."""
    orders = [order_for_insert(csv_row) for csv_row in read_csv_rows(csv_file)]
    connection.executemany(
        "INSERT INTO orders (order_id, customer_id, order_date, status, total_cents,"
        " payment_method, tracking_code, estimated_delivery, notes)"
        " VALUES (:order_id, :customer_id, :order_date, :status, :total_cents,"
        " :payment_method, :tracking_code, :estimated_delivery, :notes)",
        orders,
    )


def seed_order_items(connection: sqlite3.Connection, csv_file: Path) -> None:
    """Insert the order items from their CSV file."""
    order_items = read_csv_rows(csv_file)
    connection.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity)"
        " VALUES (:order_id, :product_id, :quantity)",
        order_items,
    )


def product_for_insert(csv_row: dict[str, str]) -> dict[str, str | int]:
    """Return a product's CSV row with its price converted to cents."""
    price_cents = brl_to_cents(csv_row["price_brl"])
    return {**csv_row, "price_cents": price_cents}


def order_for_insert(csv_row: dict[str, str]) -> dict[str, str | int | None]:
    """Return an order's CSV row with its total in cents and its empty fields as None."""
    return {
        **csv_row,
        "total_cents": brl_to_cents(csv_row["total_brl"]),
        "tracking_code": empty_to_none(csv_row["tracking_code"]),
        "estimated_delivery": empty_to_none(csv_row["estimated_delivery"]),
        "notes": empty_to_none(csv_row["notes"]),
    }


def read_csv_rows(csv_file: Path) -> list[dict[str, str]]:
    """Read a CSV file into one dict per row, keyed by the header's column names."""
    rows: list[dict[str, str]] = []
    with csv_file.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            check_field_count(row, csv_file, reader.line_num)
            rows.append(row)
    return rows


def check_field_count(
    csv_row: dict[str | None, object], csv_file: Path, line_number: int
) -> None:
    """Raise ValueError if a CSV row has more or fewer fields than the header."""
    has_extra_fields = None in csv_row
    has_missing_fields = None in csv_row.values()
    if has_extra_fields or has_missing_fields:
        raise ValueError(
            f"{csv_file.name}, line {line_number}: the number of fields differs from the header."
        )


def brl_to_cents(amount_brl: str) -> int:
    """Convert an amount in reais written as text, such as "599.9", to whole cents."""
    try:
        amount = Decimal(amount_brl)
    except InvalidOperation as error:
        raise ValueError(f"Amount {amount_brl!r} is not a number.") from error
    if not amount.is_finite():
        raise ValueError(f"Amount {amount_brl!r} is not a finite number.")
    cents = amount * 100
    if cents != cents.to_integral_value():
        raise ValueError(f"Amount {amount_brl!r} has a fraction of a cent.")
    return int(cents)


def empty_to_none(value: str) -> str | None:
    """Return None for an empty CSV field, so the database stores NULL instead of ''."""
    if not value.strip():
        return None
    return value
