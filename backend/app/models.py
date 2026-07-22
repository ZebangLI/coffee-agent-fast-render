from pydantic import BaseModel, Field


class Location(BaseModel):
    lat: float = 40.731
    lng: float = -73.992


class ChatRequest(BaseModel):
    user_id: str = "u_001"
    message: str
    location: Location = Field(default_factory=Location)


class DrinkIntent(BaseModel):
    drink: str
    temperature: str | None = None
    size: str = "medium"


class Recommendation(BaseModel):
    shop_id: str
    shop_name: str
    product_id: str
    product_name: str
    price: float
    distance_km: float
    wait_minutes: int
    score: float


class ChatResponse(BaseModel):
    intent: DrinkIntent
    recommendations: list[Recommendation]


class CreateOrderRequest(BaseModel):
    user_id: str = "u_001"
    product_id: str
    quantity: int = Field(default=1, ge=1, le=10)
    idempotency_key: str = Field(min_length=8)


class OrderResponse(BaseModel):
    order_id: str
    status: str
    shop_id: str
    product_id: str
    quantity: int
    total: float
    payment_status: str
    tx_hash: str
    explorer_url: str | None = None
    virtual_card_last4: str | None = None


class UpdateInventoryRequest(BaseModel):
    inventory: int = Field(ge=0, le=10000)
