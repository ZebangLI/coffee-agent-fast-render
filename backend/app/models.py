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


class VoiceChatResponse(ChatResponse):
    transcript: str


class TranscriptionResponse(BaseModel):
    transcript: str


class SelectionRequest(BaseModel):
    message: str
    option_count: int = Field(ge=1, le=10)


class SelectionResponse(BaseModel):
    selected_index: int | None = None


class CreateOrderRequest(BaseModel):
    user_id: str = "u_001"
    product_id: str
    quantity: int = Field(default=1, ge=1, le=10)
    idempotency_key: str = Field(min_length=8)
    buyer_email: str | None = None
    buyer_api_key: str | None = None


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
    approval_id: str | None = None


class UpdateInventoryRequest(BaseModel):
    inventory: int = Field(ge=0, le=10000)


class AigenticRegisterRequest(BaseModel):
    email: str
    password: str
    address: str = "New York, NY"


class AigenticRegisterResponse(BaseModel):
    email: str
    api_key: str


class AigenticLoginRequest(BaseModel):
    email: str
    password: str


class AigenticLoginResponse(BaseModel):
    email: str
    ok: bool
    api_key: str | None = None
