from __future__ import annotations

import os
import sqlite3
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

from .models import Location, OrderResponse, Recommendation

DB_PATH = Path(__file__).resolve().parents[2] / "coffee_agent_fast.db"


class PostgresConnection:
    def __init__(self, database_url: str) -> None:
        import psycopg
        from psycopg.rows import dict_row

        self._conn = psycopg.connect(database_url, row_factory=dict_row)

    def __enter__(self) -> "PostgresConnection":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        self._conn.close()

    def execute(self, sql: str, params: tuple[Any, ...] | None = None):
        return self._conn.execute(sql.replace("?", "%s"), params or ())

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        with self._conn.cursor() as cursor:
            cursor.executemany(sql.replace("?", "%s"), rows)


def using_postgres() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def connect() -> sqlite3.Connection | PostgresConnection:
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return PostgresConnection(database_url)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shops (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                provider TEXT NOT NULL,
                address TEXT NOT NULL,
                lat REAL NOT NULL,
                lng REAL NOT NULL,
                wait_minutes INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                shop_id TEXT NOT NULL,
                name TEXT NOT NULL,
                aliases TEXT NOT NULL,
                price REAL NOT NULL,
                inventory INTEGER NOT NULL,
                FOREIGN KEY (shop_id) REFERENCES shops(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                shop_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                status TEXT NOT NULL,
                total REAL NOT NULL,
                payment_status TEXT NOT NULL,
                tx_hash TEXT NOT NULL,
                explorer_url TEXT,
                virtual_card_last4 TEXT,
                idempotency_key TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (shop_id) REFERENCES shops(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
            """
        )
        conn.executemany(
            _upsert_sql(
                "shops",
                ("id", "name", "provider", "address", "lat", "lng", "wait_minutes"),
                ("name", "provider", "address", "lat", "lng", "wait_minutes"),
            ),
            [
                ("starbucks_001", "Starbucks Broadway", "starbucks", "Broadway, New York, NY", 40.7301, -73.9912, 8),
                ("local_cafe_001", "Local Cafe Washington Square", "local_cafe", "Washington Square, New York, NY", 40.7308, -73.9973, 10),
                ("campus_cafe_001", "Campus Cafe NYU", "campus_cafe", "NYU Campus, New York, NY", 40.7295, -73.9965, 6),
            ],
        )
        conn.executemany(
            _upsert_sql(
                "products",
                ("id", "shop_id", "name", "aliases", "price", "inventory"),
                ("shop_id", "name", "aliases", "price", "inventory"),
            ),
            [
                ("starbucks_latte", "starbucks_001", "Medium Latte", "latte,coffee with milk", 5.25, 20),
                ("starbucks_americano", "starbucks_001", "Medium Americano", "americano,black coffee", 4.10, 30),
                ("starbucks_cold_brew", "starbucks_001", "Medium Cold Brew", "cold brew,iced coffee", 4.75, 15),
                ("local_latte", "local_cafe_001", "House Latte", "latte,coffee with milk", 4.80, 18),
                ("local_americano", "local_cafe_001", "House Americano", "americano,black coffee", 3.90, 20),
                ("local_cold_brew", "local_cafe_001", "Small Batch Cold Brew", "cold brew,iced coffee", 4.60, 12),
                ("campus_latte", "campus_cafe_001", "Campus Latte", "latte,coffee with milk", 4.50, 25),
                ("campus_americano", "campus_cafe_001", "Campus Americano", "americano,black coffee", 3.50, 25),
                ("campus_cold_brew", "campus_cafe_001", "Campus Cold Brew", "cold brew,iced coffee", 4.25, 10),
            ],
        )


def recommend_products(drink: str, location: Location) -> list[Recommendation]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT products.id AS product_id, products.name AS product_name,
                   products.aliases, products.price, products.inventory,
                   shops.id AS shop_id, shops.name AS shop_name, shops.lat, shops.lng, shops.wait_minutes
            FROM products
            JOIN shops ON shops.id = products.shop_id
            WHERE LOWER(products.aliases) LIKE ?
              AND products.inventory > 0
            """,
            (f"%{drink.lower()}%",),
        ).fetchall()

    recommendations = []
    for row in rows:
        distance = distance_km(location, row["lat"], row["lng"])
        score = round(
            max(0, 1 - distance / 3) * 0.4
            + max(0, 1 - row["wait_minutes"] / 30) * 0.3
            + max(0, 1 - row["price"] / 10) * 0.3,
            4,
        )
        recommendations.append(
            Recommendation(
                shop_id=row["shop_id"],
                shop_name=row["shop_name"],
                product_id=row["product_id"],
                product_name=row["product_name"],
                price=row["price"],
                distance_km=round(distance, 2),
                wait_minutes=row["wait_minutes"],
                score=score,
            )
        )
    return sorted(recommendations, key=lambda item: item.score, reverse=True)


def list_orders(limit: int = 20) -> list[OrderResponse]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, shop_id, product_id, quantity, status, total,
                   payment_status, tx_hash, explorer_url, virtual_card_last4
            FROM orders
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_order_from_row(row) for row in rows]


def list_shops() -> list[dict]:
    with connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM shops ORDER BY name").fetchall()]


def list_shop_products(shop_id: str) -> list[dict]:
    with connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM products WHERE shop_id = ? ORDER BY name",
                (shop_id,),
            ).fetchall()
        ]


def list_shop_orders(shop_id: str) -> list[dict]:
    with connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT orders.*, products.name AS product_name, shops.name AS shop_name
                FROM orders
                JOIN products ON products.id = orders.product_id
                JOIN shops ON shops.id = products.shop_id
                WHERE products.shop_id = ?
                ORDER BY orders.created_at DESC
                LIMIT 50
                """,
                (shop_id,),
            ).fetchall()
        ]


def update_inventory(product_id: str, inventory: int) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT id FROM products WHERE id = ?", (product_id,)).fetchone()
        if row is None:
            raise ValueError("Product not found")
        conn.execute("UPDATE products SET inventory = ? WHERE id = ?", (inventory, product_id))
    return {"product_id": product_id, "inventory": inventory}


def insert_order(order: OrderResponse, user_id: str, idempotency_key: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO orders
                (id, user_id, shop_id, product_id, quantity, status, total,
                 payment_status, tx_hash, explorer_url, virtual_card_last4, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.order_id,
                user_id,
                order.shop_id,
                order.product_id,
                order.quantity,
                order.status,
                order.total,
                order.payment_status,
                order.tx_hash,
                order.explorer_url,
                order.virtual_card_last4,
                idempotency_key,
            ),
        )
        conn.execute(
            "UPDATE products SET inventory = inventory - ? WHERE id = ?",
            (order.quantity, order.product_id),
        )


def get_order_by_idempotency_key(idempotency_key: str) -> OrderResponse | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, user_id, shop_id, product_id, quantity, status, total,
                   payment_status, tx_hash, explorer_url, virtual_card_last4
            FROM orders
            WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchone()
    return _order_from_row(row) if row else None


def get_product_for_order(product_id: str) -> dict:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT products.*, shops.name AS shop_name
            FROM products
            JOIN shops ON shops.id = products.shop_id
            WHERE products.id = ?
            """,
            (product_id,),
        ).fetchone()
    if row is None:
        raise ValueError("Product not found")
    return dict(row)


def _order_from_row(row: Any) -> OrderResponse:
    return OrderResponse(
        order_id=row["id"],
        status=row["status"],
        shop_id=row["shop_id"],
        product_id=row["product_id"],
        quantity=row["quantity"],
        total=row["total"],
        payment_status=row["payment_status"],
        tx_hash=row["tx_hash"],
        explorer_url=row["explorer_url"],
        virtual_card_last4=row["virtual_card_last4"],
    )


def distance_km(location: Location, lat: float, lng: float) -> float:
    radius = 6371
    dlat = radians(lat - location.lat)
    dlng = radians(lng - location.lng)
    lat1 = radians(location.lat)
    lat2 = radians(lat)
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return 2 * radius * asin(sqrt(h))


def _upsert_sql(
    table: str,
    columns: tuple[str, ...],
    update_columns: tuple[str, ...],
    conflict_columns: tuple[str, ...] = ("id",),
) -> str:
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)
    if not using_postgres():
        return f"INSERT OR REPLACE INTO {table} ({column_sql}) VALUES ({placeholders})"

    conflict_sql = ", ".join(conflict_columns)
    update_sql = ", ".join(f"{column} = EXCLUDED.{column}" for column in update_columns)
    return (
        f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_sql}) DO UPDATE SET {update_sql}"
    )
