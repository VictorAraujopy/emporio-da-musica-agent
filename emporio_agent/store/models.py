"""Typed records of the store's data, built by the queries from database rows."""

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

ONE_CENT = Decimal("0.01")


class ProductStatus(StrEnum):
    """A product's sales status in the catalog."""

    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    COMING_SOON = "coming_soon"


class OrderStatus(StrEnum):
    """Where an order is in its life cycle."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ActivePromotion:
    """The promotion currently running on a product."""

    description: str
    discount_percent: int


@dataclass(frozen=True)
class Product:
    """A catalog product, with its category name and its active promotion, if any."""

    product_id: int
    name: str
    category: str
    description: str
    price_brl: Decimal
    stock_quantity: int
    status: ProductStatus
    specs: dict[str, str]
    promotion: ActivePromotion | None

    @property
    def promotional_price_brl(self) -> Decimal | None:
        """Return the price after the active promotion's discount, rounded to the cent."""
        if self.promotion is None:
            return None
        discount_percent = self.promotion.discount_percent
        fraction_to_pay = Decimal(100 - discount_percent) / 100
        discounted_price = self.price_brl * fraction_to_pay
        return discounted_price.quantize(ONE_CENT, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class OrderItem:
    """One line of an order: which product and how many units."""

    product_name: str
    quantity: int


@dataclass(frozen=True)
class Order:
    """An order's own data, without any customer information."""

    order_id: int
    order_date: date
    status: OrderStatus
    total_brl: Decimal
    payment_method: str
    tracking_code: str | None
    estimated_delivery: date | None
    notes: str | None
    items: tuple[OrderItem, ...]
