-- Store database schema: one STRICT table per CSV file in data/.

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    description TEXT    NOT NULL
) STRICT;

CREATE TABLE products (
    product_id     INTEGER PRIMARY KEY,
    name           TEXT    NOT NULL,
    category_id    INTEGER NOT NULL REFERENCES categories (category_id),
    description    TEXT    NOT NULL,
    price_cents    INTEGER NOT NULL CHECK (price_cents >= 0),
    stock_quantity INTEGER NOT NULL CHECK (stock_quantity >= 0),
    status         TEXT    NOT NULL
        CHECK (status IN ('active', 'discontinued', 'coming_soon')),
    specs          TEXT    NOT NULL CHECK (json_valid(specs) AND json_type(specs) = 'object'),
    created_at     TEXT    NOT NULL CHECK (date(created_at) IS created_at)
) STRICT;

CREATE TABLE promotions (
    promotion_id     INTEGER PRIMARY KEY,
    product_id       INTEGER NOT NULL REFERENCES products (product_id),
    discount_percent INTEGER NOT NULL CHECK (discount_percent BETWEEN 1 AND 99),
    description      TEXT    NOT NULL,
    is_active        INTEGER NOT NULL CHECK (is_active IN (0, 1))
) STRICT;

CREATE UNIQUE INDEX one_active_promotion_per_product
    ON promotions (product_id) WHERE is_active = 1;

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    phone       TEXT    NOT NULL,
    email       TEXT    NOT NULL,
    city        TEXT    NOT NULL
) STRICT;

CREATE TABLE orders (
    order_id           INTEGER PRIMARY KEY,
    customer_id        INTEGER NOT NULL REFERENCES customers (customer_id),
    order_date         TEXT    NOT NULL CHECK (date(order_date) IS order_date),
    status             TEXT    NOT NULL
        CHECK (status IN ('pending', 'confirmed', 'shipped', 'delivered', 'cancelled')),
    total_cents        INTEGER NOT NULL CHECK (total_cents >= 0),
    payment_method     TEXT    NOT NULL,
    tracking_code      TEXT,
    estimated_delivery TEXT    CHECK (date(estimated_delivery) IS estimated_delivery),
    notes              TEXT
) STRICT;

CREATE TABLE order_items (
    order_id   INTEGER NOT NULL REFERENCES orders (order_id),
    product_id INTEGER NOT NULL REFERENCES products (product_id),
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (order_id, product_id)
) STRICT;
