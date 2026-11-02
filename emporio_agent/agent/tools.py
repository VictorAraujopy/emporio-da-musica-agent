"""Tools the chat model calls: the store's data and price calculators, all returning text."""

import math
import sqlite3
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from langchain.tools import tool

from emporio_agent.store import queries
from emporio_agent.store.models import ONE_CENT, Order, Product, ProductStatus

NO_PRODUCT_FOUND = (
    "No product matches this search. Product names rarely contain the instrument type, and"
    " a category must be an exact name: check them with list_categories and search again."
)
NO_PROMOTION_ACTIVE = "No promotion is active right now."
ORDER_NOT_FOUND = "No order matches this order number and email together."
STORE_DATA_UNAVAILABLE = "The store's data is unavailable right now: say you could not check it."
MAX_PRICE_BRL = 1_000_000_000
MAX_INSTALLMENTS = 60
INVALID_DISCOUNT = "Not computed: the price must be real and the percent between 0 and 100."
INVALID_INSTALLMENTS = "Not computed: the total must be real and installments from 1 to 60."


@tool
def search_catalog(
    name: str | None = None,
    category: str | None = None,
    max_price_brl: float | None = None,
) -> str:
    """Search products by part of the name, an exact category name and/or a price limit."""
    max_price = Decimal(str(max_price_brl)) if max_price_brl is not None else None
    try:
        products = queries.search_products(name=name, category=category, max_price_brl=max_price)
    except sqlite3.Error:
        return STORE_DATA_UNAVAILABLE
    if not products:
        return NO_PRODUCT_FOUND
    return describe_products(products)


@tool
def list_promotions() -> str:
    """List the products on promotion right now, with their discounted prices."""
    try:
        products = queries.find_products_on_promotion()
    except sqlite3.Error:
        return STORE_DATA_UNAVAILABLE
    if not products:
        return NO_PROMOTION_ACTIVE
    return describe_products(products)


@tool
def list_categories() -> str:
    """List the names of the store's product categories."""
    try:
        category_names = queries.list_category_names()
    except sqlite3.Error:
        return STORE_DATA_UNAVAILABLE
    return ", ".join(category_names)


@tool
def look_up_order(order_id: int, email: str) -> str:
    """Look up an order; needs both its number and the email on the customer's record."""
    try:
        order = queries.find_order(order_id, email)
    except OverflowError:
        return ORDER_NOT_FOUND
    except sqlite3.Error:
        return STORE_DATA_UNAVAILABLE
    if order is None:
        return ORDER_NOT_FOUND
    return describe_order(order)


@tool
def apply_discount(price_brl: float, discount_percent: float) -> str:
    """Compute a price after a percent discount, such as one stated in the store policies."""
    if not is_valid_discount(price_brl, discount_percent):
        return INVALID_DISCOUNT
    price = Decimal(str(price_brl))
    fraction_to_pay = (100 - Decimal(str(discount_percent))) / 100
    discounted_price = (price * fraction_to_pay).quantize(ONE_CENT, rounding=ROUND_HALF_UP)
    return f"{format_brl(price)} with {discount_percent:g}% off: {format_brl(discounted_price)}"


@tool
def split_into_installments(total_brl: float, installments: int) -> str:
    """Compute the installment values of a total; the first absorbs the rounding difference."""
    if not is_valid_price(total_brl) or not 1 <= installments <= MAX_INSTALLMENTS:
        return INVALID_INSTALLMENTS
    total = Decimal(str(total_brl))
    regular_value = (total / installments).quantize(ONE_CENT, rounding=ROUND_HALF_UP)
    first_value = total - regular_value * (installments - 1)
    split_heading = f"{format_brl(total)} in {installments}x"
    if first_value == regular_value:
        return f"{split_heading}: {installments}x {format_brl(regular_value)}"
    other_installments = f"{installments - 1}x {format_brl(regular_value)}"
    return f"{split_heading}: 1x {format_brl(first_value)} + {other_installments}"


AGENT_TOOLS = [
    search_catalog,
    list_promotions,
    list_categories,
    look_up_order,
    apply_discount,
    split_into_installments,
]


def is_valid_discount(price_brl: float, discount_percent: float) -> bool:
    """Accept a real price and a percent strictly between 0 and 100."""
    return is_valid_price(price_brl) and 0 < discount_percent < 100


def is_valid_price(price_brl: float) -> bool:
    """Accept a finite price from 0 up to a ceiling no store price reaches."""
    return math.isfinite(price_brl) and 0 <= price_brl <= MAX_PRICE_BRL


def describe_products(products: list[Product]) -> str:
    """Write one product per line."""
    return "\n".join(describe_product(product) for product in products)


def describe_product(product: Product) -> str:
    """Write a product as labeled fields: prices, promotion, stock, status, specs, description."""
    fields = [
        product.name,
        f"category: {product.category}",
        f"price: {format_brl(product.price_brl)}",
        f"promotion: {describe_promotion(product)}",
        f"stock: {product.stock_quantity}",
        f"status: {describe_availability(product)}",
        f"specs: {describe_specs(product.specs)}",
        f"description: {product.description}",
    ]
    return " | ".join(fields)


def describe_promotion(product: Product) -> str:
    """Write the active promotion and the promotional price, or 'none'."""
    if product.promotion is None or product.promotional_price_brl is None:
        return "none"
    promotional_price = format_brl(product.promotional_price_brl)
    promotion = product.promotion
    return f"{promotion.discount_percent}% off ({promotion.description}), now {promotional_price}"


def describe_availability(product: Product) -> str:
    """Say whether the customer can buy the product now, and if not, why."""
    if product.status == ProductStatus.DISCONTINUED:
        return "discontinued"
    if product.status == ProductStatus.COMING_SOON:
        return "coming soon"
    if product.is_available:
        return "available"
    return "out of stock"


def describe_specs(specs: dict[str, str]) -> str:
    """Write the specs as 'key: value' pairs."""
    return "; ".join(f"{key}: {value}" for key, value in specs.items())


def describe_order(order: Order) -> str:
    """Write an order's own fields and items; nothing about the customer."""
    items = ", ".join(f"{item.quantity}x {item.product_name}" for item in order.items)
    fields = [
        f"order {order.order_id}",
        f"ordered on: {format_date(order.order_date)}",
        f"status: {order.status}",
        f"total: {format_brl(order.total_brl)}",
        f"payment: {order.payment_method}",
        f"tracking code: {order.tracking_code or 'none yet'}",
        f"estimated delivery: {format_optional_date(order.estimated_delivery)}",
        f"notes: {order.notes or 'none'}",
        f"items: {items}",
    ]
    return " | ".join(fields)


def format_brl(amount: Decimal) -> str:
    """Write an amount the Brazilian way: R$ 1.234,56."""
    us_style = f"{amount:,.2f}"
    brazilian_style = us_style.replace(",", " ").replace(".", ",").replace(" ", ".")
    return f"R$ {brazilian_style}"


def format_date(day: date) -> str:
    """Write a date the Brazilian way: 28/02/2026."""
    return day.strftime("%d/%m/%Y")


def format_optional_date(day: date | None) -> str:
    """Write a date the Brazilian way, or 'not set' when there is none."""
    if day is None:
        return "not set"
    return format_date(day)
