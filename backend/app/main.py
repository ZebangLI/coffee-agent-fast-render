from __future__ import annotations

import os
from html import escape
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .database import (
    get_order_by_idempotency_key,
    get_product_for_order,
    init_db,
    insert_order,
    list_orders,
    list_shop_orders,
    list_shop_products,
    list_shops,
    recommend_products,
    update_inventory,
)
from .llm import decide_agent_action, llm_status, parse_intent, parse_selection, transcribe_audio
from .models import (
    AgentRequest,
    AgentResponse,
    ChatRequest,
    ChatResponse,
    AigenticLoginRequest,
    AigenticLoginResponse,
    AigenticRegisterRequest,
    AigenticRegisterResponse,
    CreateOrderRequest,
    DrinkIntent,
    Location,
    OrderResponse,
    SelectionRequest,
    SelectionResponse,
    TranscriptionResponse,
    UpdateInventoryRequest,
    VoiceChatResponse,
)
from .payment import ext_login, ext_register, record_payment

app = FastAPI(title="Coffee Agent Fast Render", version="0.2.0")

UNAVAILABLE_PRODUCT_MESSAGE = (
    "Sorry, this demo can only order coffee right now. "
    "That product is not available yet."
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/llm/status")
def api_llm_status() -> dict:
    return llm_status()


@app.post("/api/aigenticpay/register", response_model=AigenticRegisterResponse)
def aigenticpay_register(request: AigenticRegisterRequest) -> AigenticRegisterResponse:
    try:
        result = ext_register(request.email, request.password, request.address)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    api_key = result.get("api_key")
    if not api_key:
        raise HTTPException(status_code=502, detail="AigenticPay did not return an api_key.")
    return AigenticRegisterResponse(email=request.email, api_key=str(api_key))


@app.post("/api/aigenticpay/login", response_model=AigenticLoginResponse)
def aigenticpay_login(request: AigenticLoginRequest) -> AigenticLoginResponse:
    result = ext_login(request.email, request.password)
    if not result.get("ok"):
        return AigenticLoginResponse(email=request.email, ok=False)
    api_key = result.get("api_key")
    if not api_key:
        raise HTTPException(status_code=502, detail="AigenticPay login did not return an api_key.")
    return AigenticLoginResponse(email=request.email, ok=True, api_key=str(api_key))


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return _chat_from_text(request.message, request.location)


@app.post("/api/agent", response_model=AgentResponse)
def agent(request: AgentRequest) -> AgentResponse:
    decision = decide_agent_action(request.message, request.context)
    action = decision.get("action") or "chat"

    if action == "search":
        intent = DrinkIntent(
            drink=decision.get("drink") or "americano",
            quantity=decision.get("quantity") or 1,
        )
        recommendations = recommend_products(intent.drink, request.location)
        if not recommendations:
            return AgentResponse(
                action="unsupported",
                message=UNAVAILABLE_PRODUCT_MESSAGE,
            )
        return AgentResponse(
            action="show_options",
            intent=intent,
            recommendations=recommendations,
        )

    return AgentResponse(
        action=action,
        message=decision.get("message"),
        selected_index=decision.get("selected_index"),
        quantity=decision.get("quantity"),
        product_id=decision.get("product_id"),
    )


@app.post("/api/selection", response_model=SelectionResponse)
def selection(request: SelectionRequest) -> SelectionResponse:
    selected_index, quantity = parse_selection(request.message, request.option_count)
    return SelectionResponse(selected_index=selected_index, quantity=quantity)


@app.post("/api/voice/transcribe", response_model=TranscriptionResponse)
async def voice_transcribe(audio: UploadFile = File(...)) -> TranscriptionResponse:
    transcript = await _transcribe_upload(audio)
    return TranscriptionResponse(transcript=transcript)


@app.post("/api/voice/chat", response_model=VoiceChatResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    user_id: str = Form("u_001"),
    lat: float = Form(40.731),
    lng: float = Form(-73.992),
) -> VoiceChatResponse:
    del user_id
    transcript = await _transcribe_upload(audio)
    try:
        chat_response = _chat_from_text(transcript, Location(lat=lat, lng=lng))
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": UNAVAILABLE_PRODUCT_MESSAGE,
                    "transcript": transcript,
                },
            ) from exc
        raise
    return VoiceChatResponse(
        transcript=transcript,
        intent=chat_response.intent,
        recommendations=chat_response.recommendations,
    )


@app.get("/api/orders", response_model=list[OrderResponse])
def api_list_orders(user_id: str | None = None) -> list[OrderResponse]:
    return list_orders(user_id=user_id)


@app.post("/api/orders", response_model=OrderResponse)
def create_order(request: CreateOrderRequest) -> OrderResponse:
    existing = get_order_by_idempotency_key(request.idempotency_key)
    if existing:
        return existing

    try:
        product = get_product_for_order(request.product_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if product["inventory"] < request.quantity:
        raise HTTPException(status_code=409, detail="Not enough inventory")

    total = round(product["price"] * request.quantity, 2)
    order_id = f"ord_{uuid4().hex[:10]}"
    request_id = f"REQ-{uuid4().hex[:10].upper()}"
    try:
        payment = record_payment(
            {
                "source_app": "coffee-agent-fast-render",
                "external_order_id": order_id,
                "user_ref": request.user_id,
                "request_id": request_id,
                "shop_id": product["shop_id"],
                "shop_name": product["shop_name"],
                "merchant_id": product.get("merchant_id") or "00001",
                "mcc_code": product.get("mcc_code") or "5814",
                "product_id": product["id"],
                "product_name": product["name"],
                "quantity": request.quantity,
                "unit_price": product["price"],
                "amount": total,
                "currency": "USD",
                "buyer_email": request.buyer_email,
                "buyer_api_key": request.buyer_api_key,
                "idempotency_key": request.idempotency_key,
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    order = OrderResponse(
        order_id=order_id,
        status="confirmed",
        shop_id=product["shop_id"],
        product_id=product["id"],
        quantity=request.quantity,
        total=total,
        payment_status=payment["payment_status"],
        tx_hash=payment["tx_hash"],
        explorer_url=payment.get("explorer_url"),
        virtual_card_last4=payment.get("virtual_card_last4"),
        approval_id=payment.get("approval_id"),
    )
    insert_order(order, request.user_id, request.idempotency_key)
    return order


@app.get("/api/merchant/shops")
def api_shops() -> list[dict]:
    return list_shops()


@app.get("/api/merchant/shops/{shop_id}/products")
def api_shop_products(shop_id: str) -> list[dict]:
    return list_shop_products(shop_id)


@app.get("/api/merchant/shops/{shop_id}/orders")
def api_shop_orders(shop_id: str) -> list[dict]:
    return list_shop_orders(shop_id)


@app.post("/api/merchant/products/{product_id}/inventory")
def api_update_inventory(product_id: str, request: UpdateInventoryRequest) -> dict:
    try:
        return update_inventory(product_id, request.inventory)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _chat_from_text(message: str, location: Location) -> ChatResponse:
    intent = parse_intent(message)
    if intent is None:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_PRODUCT_MESSAGE)
    recommendations = recommend_products(intent.drink, location)
    return ChatResponse(intent=intent, recommendations=recommendations)


async def _transcribe_upload(audio: UploadFile) -> str:
    audio_bytes = await audio.read()
    max_bytes = int(os.environ.get("MAX_VOICE_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio received.")
    if len(audio_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail="Audio file is too large.")

    transcript = transcribe_audio(
        audio_bytes,
        audio.filename or "voice.webm",
        audio.content_type or "audio/webm",
    )
    if not transcript:
        raise HTTPException(status_code=503, detail="Speech transcription failed.")
    return transcript


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Coffee Agent Fast</title>
  <style>
    :root { --bg:#f6f3ee; --panel:#fff; --text:#1f2421; --muted:#66736b; --line:#ded8cf; --accent:#176b54; --soft:#eef6f1; --danger:#9b1c1c; }
    * { box-sizing:border-box; }
    body { margin:0; height:100vh; overflow:hidden; background:var(--bg); color:var(--text); font-family:Arial,sans-serif; }
    header { background:#20352d; color:white; padding:14px 20px; display:flex; justify-content:space-between; align-items:center; }
    header h1 { margin:0; font-size:20px; }
    main { height:calc(100vh - 54px); max-width:1180px; margin:0 auto; padding:18px; display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:16px; }
    .hidden { display:none !important; }
    .auth-view { height:calc(100vh - 54px); display:grid; place-items:center; padding:24px; }
    .auth-card { width:min(480px, 100%); background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:22px; display:grid; gap:14px; }
    .auth-card h2 { margin:0; font-size:24px; }
    .auth-tabs { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .auth-tabs button { background:#efe8dd; color:#20352d; border:1px solid #d8d0c7; }
    .auth-tabs button.active { background:var(--accent); color:white; border-color:var(--accent); }
    .auth-form { display:grid; gap:10px; }
    .auth-note { color:var(--muted); font-size:13px; line-height:1.45; }
    .auth-status { color:var(--muted); font-size:13px; min-height:18px; }
    section { min-height:0; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    .head { padding:13px 16px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; align-items:center; }
    .chat, .orders { display:grid; grid-template-rows:auto 1fr auto; }
    .orders { grid-template-rows:auto 1fr; }
    .log, .orders-list { overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:12px; }
    .msg { max-width:82%; padding:12px 14px; border-radius:8px; line-height:1.45; }
    .agent { background:var(--soft); border:1px solid #d3e7dc; align-self:flex-start; }
    .user { background:var(--accent); color:white; align-self:flex-end; }
    .error { background:#fff1f1; border:1px solid #f0b9b9; color:var(--danger); }
    .composer { border-top:1px solid var(--line); padding:12px; display:grid; grid-template-columns:1fr 78px 90px; gap:10px; }
    textarea, input { border:1px solid #cfc7bd; border-radius:6px; padding:10px; font:inherit; }
    button { border:0; border-radius:6px; padding:10px 12px; background:var(--accent); color:white; font-weight:700; cursor:pointer; }
    button.secondary { background:#efe8dd; color:#20352d; border:1px solid #d8d0c7; }
    button.recording { background:#9b1c1c; color:white; }
    .cards { display:grid; gap:10px; margin-top:10px; }
    .card, .order-card { border:1px solid #d8d0c7; border-radius:8px; background:#fffefa; padding:12px; }
    .order-card { background:#f4fbf7; border-color:#cbded5; font-size:13px; line-height:1.45; }
    .top { display:flex; justify-content:space-between; gap:10px; }
    .muted { color:var(--muted); font-size:13px; }
    .pill { display:inline-block; margin:8px 6px 0 0; padding:4px 7px; border:1px solid #d9d0c6; border-radius:999px; font-size:12px; }
    .account-chip { color:#dfeee8; font-size:13px; display:flex; gap:10px; align-items:center; }
    .link-button { background:transparent; color:white; border:1px solid rgba(255,255,255,.45); padding:6px 8px; font-size:12px; }
  </style>
</head>
<body>
<header><h1>Coffee Agent</h1><span id="account-chip" class="account-chip"></span></header>
<section id="auth" class="auth-view">
  <div class="auth-card">
    <h2>AigenticPay Account</h2>
    <p class="auth-note">Register or login through AigenticPay. The buyer API key is saved locally after a successful response.</p>
    <div class="auth-tabs">
      <button id="tab-register" class="active" onclick="setAuthMode('register')">Register</button>
      <button id="tab-login" onclick="setAuthMode('login')">Login</button>
    </div>
    <div class="auth-form">
      <input id="auth-email" placeholder="Email">
      <input id="auth-password" type="password" placeholder="Password">
      <input id="auth-address" placeholder="Address" value="New York, NY">
      <input id="auth-api-key" class="hidden" placeholder="Buyer API key">
      <button id="auth-submit" onclick="submitAuth()">Register and get API key</button>
      <div id="auth-status" class="auth-status"></div>
    </div>
  </div>
</section>
<main id="app" class="hidden">
  <section class="chat">
    <div class="head"><strong>Chat</strong><span></span></div>
    <div id="log" class="log"><div class="msg agent">Hi, tell me what coffee you want.</div></div>
    <div class="composer">
      <textarea id="message" placeholder="I want a latte near me"></textarea>
      <button id="voice" class="secondary" title="Record voice">Voice</button>
      <button id="send">Send</button>
    </div>
  </section>
  <section class="orders">
    <div class="head"><strong>Orders</strong><button onclick="loadOrders()">Refresh</button></div>
    <div id="orders" class="orders-list"></div>
  </section>
</main>
<script>
let latest = [];
let pendingQuantity = 1;
let activeCardsMessage = null;
let lastOrder = null;
let recorder = null;
let chunks = [];
let recording = false;
let orderInFlight = false;
let apAccount = JSON.parse(localStorage.getItem("coffeeAgentApAccount") || "{}");
let authMode = "register";
const log = document.getElementById("log");
function add(role, html){ const n=document.createElement("div"); n.className=`msg ${role}`; n.innerHTML=html; log.appendChild(n); log.scrollTop=log.scrollHeight; return n; }
function clearAuthInputs(){
  document.getElementById("auth-email").value = "";
  document.getElementById("auth-password").value = "";
  document.getElementById("auth-address").value = "New York, NY";
  document.getElementById("auth-api-key").value = "";
}
function apiErrorMessage(payload, text, status){
  const detail = payload && payload.detail;
  if(detail && typeof detail === "object") return detail.message || JSON.stringify(detail);
  return detail || text || `Request failed: ${status}`;
}
async function fetchWithTimeout(path, options={}, timeoutMs=45000){
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try{
    return await fetch(path, {...options, signal: controller.signal});
  }catch(e){
    if(e.name === "AbortError") throw new Error("Request timed out. The cloud LLM or AigenticPay API is responding slowly.");
    throw e;
  }finally{
    clearTimeout(timer);
  }
}
async function api(path, options={}){
  const r=await fetchWithTimeout(path,{headers:{"Content-Type":"application/json"},...options});
  const t=await r.text();
  const d=t?JSON.parse(t):{};
  if(!r.ok) throw new Error(apiErrorMessage(d,t,r.status));
  return d;
}
async function apiForm(path, form){
  const r=await fetchWithTimeout(path,{method:"POST",body:form},60000);
  const t=await r.text();
  const d=t?JSON.parse(t):{};
  if(!r.ok){
    const detail=d.detail;
    const message=detail && typeof detail === "object" ? detail.message : (detail || t || r.status);
    const err=new Error(message);
    if(detail && typeof detail === "object") err.transcript=detail.transcript;
    throw err;
  }
  return d;
}
function renderRecs(data){
  latest=data.recommendations||[];
  pendingQuantity = Math.max(1, Math.min(Number(data.intent.quantity || 1), 10));
  if(!latest.length){ add("agent","No matching coffee nearby."); return; }
  activeCardsMessage = add("agent", `Intent: <strong>${data.intent.drink}</strong><span class="quantity-label">${pendingQuantity > 1 ? ` x ${pendingQuantity}` : ""}</span><div class="cards">${latest.map((x,i)=>`
    <div class="card"><div class="top"><strong>${i+1}. ${x.shop_name}</strong><strong>$${Number(x.price).toFixed(2)} each</strong></div>
    <div class="muted">${x.product_name}</div><div class="muted total-line" data-unit-price="${Number(x.price)}">Total $${(Number(x.price) * pendingQuantity).toFixed(2)}</div><span class="pill">${x.distance_km} km</span><span class="pill">${x.wait_minutes} min</span><span class="pill">score ${x.score}</span>
    <div><button data-order-button="true" onclick="order(${i})">Order ${pendingQuantity > 1 ? pendingQuantity : "this"}</button></div></div>`).join("")}</div>`);
}
function updatePendingQuantity(quantity){
  pendingQuantity = Math.max(1, Math.min(Number(quantity), 10));
  if(!activeCardsMessage) return;
  activeCardsMessage.querySelectorAll(".quantity-label").forEach(el => {
    el.textContent = pendingQuantity > 1 ? ` x ${pendingQuantity}` : "";
  });
  activeCardsMessage.querySelectorAll("[data-order-button='true']").forEach(button => {
    button.textContent = `Order ${pendingQuantity > 1 ? pendingQuantity : "this"}`;
  });
  activeCardsMessage.querySelectorAll(".total-line").forEach(line => {
    const unitPrice = Number(line.dataset.unitPrice || 0);
    line.textContent = `Total $${(unitPrice * pendingQuantity).toFixed(2)}`;
  });
}
function looksLikeProductRequest(message){
  const text = message.toLowerCase();
  return /(coffee|latte|americano|cold brew|espresso|mocha|cappuccino|tea|burger|hamburger|pizza|sandwich|buy|order|want|get|\u5496\u5561|\u62ff\u94c1|\u7f8e\u5f0f|\u51b7\u8403|\u51b0\u5496\u5561|\u6c49\u5821|\u62ab\u8428|\u4e09\u660e\u6cbb|\u5976\u8336|\u60f3\u8981|\u8981\u4e00|\u4e70)/.test(text);
}
function setAuthMode(mode){
  authMode = mode;
  document.getElementById("tab-register").classList.toggle("active", mode === "register");
  document.getElementById("tab-login").classList.toggle("active", mode === "login");
  document.getElementById("auth-address").classList.toggle("hidden", mode !== "register");
  document.getElementById("auth-api-key").classList.add("hidden");
  document.getElementById("auth-submit").textContent = mode === "register" ? "Register and get API key" : "Login";
  document.getElementById("auth-status").textContent = mode === "register"
    ? "AigenticPay will return a buyer API key after registration."
    : "AigenticPay login will return and save the buyer API key.";
}
function hasPaymentIdentity(){
  return Boolean(apAccount.email && apAccount.api_key);
}
function renderAuthState(){
  const authenticated = hasPaymentIdentity();
  document.getElementById("auth").classList.toggle("hidden", authenticated);
  document.getElementById("app").classList.toggle("hidden", !authenticated);
  document.getElementById("account-chip").innerHTML = authenticated
    ? `<span>AigenticPay: ${apAccount.email}</span><button class="link-button" onclick="signOut()">Sign out</button>`
    : "<span>Register or login to buy coffee</span>";
  if(!authenticated){
    setAuthMode(authMode);
    document.getElementById("orders").innerHTML="<span class='muted'>Sign in to see orders.</span>";
  }else{
    loadOrders();
  }
}
function signOut(){
  apAccount = {};
  localStorage.removeItem("coffeeAgentApAccount");
  clearAuthInputs();
  latest = [];
  pendingQuantity = 1;
  activeCardsMessage = null;
  lastOrder = null;
  document.querySelectorAll(".cards").forEach(c=>c.closest(".msg").remove());
  renderAuthState();
}
async function submitAuth(){
  const email=document.getElementById("auth-email").value.trim();
  const password=document.getElementById("auth-password").value;
  const address=document.getElementById("auth-address").value.trim() || "New York, NY";
  const status=document.getElementById("auth-status");
  if(!email || !password){ status.textContent="Email and password are required."; return; }
  status.textContent = authMode === "register" ? "Registering with AigenticPay..." : "Checking login with AigenticPay...";
  try{
    if(authMode === "register"){
      const data=await api("/api/aigenticpay/register",{method:"POST",body:JSON.stringify({email,password,address})});
      apAccount={email:data.email,api_key:data.api_key};
      localStorage.setItem("coffeeAgentApAccount", JSON.stringify(apAccount));
      clearAuthInputs();
      renderAuthState();
      add("agent","AigenticPay registration is ready. You can ask for coffee now.");
      return;
    }
    const data=await api("/api/aigenticpay/login",{method:"POST",body:JSON.stringify({email,password})});
    if(!data.ok){ status.textContent="Login failed."; return; }
    if(!data.api_key){
      status.textContent="Login verified, but AigenticPay did not return a buyer API key.";
      return;
    }
    apAccount={email:data.email,api_key:data.api_key};
    localStorage.setItem("coffeeAgentApAccount", JSON.stringify(apAccount));
    clearAuthInputs();
    renderAuthState();
    add("agent","AigenticPay login is ready. You can ask for coffee now.");
  }catch(e){ status.textContent=e.message; }
}
async function handleUserMessage(message, label){
  add("user", label || message);
  const thinking = add("agent","Thinking with context...");
  try{
    const data = await api("/api/agent",{
      method:"POST",
      body:JSON.stringify({
        message,
        context:{
          options:latest,
          pending_quantity:pendingQuantity,
          last_order:lastOrder
        }
      })
    });
    thinking.remove();

    if(data.action === "show_options"){
      renderRecs(data);
      return;
    }
    if(data.action === "select_option"){
      if(data.quantity !== null && data.quantity !== undefined) updatePendingQuantity(data.quantity);
      await order(data.selected_index);
      return;
    }
    if(data.action === "update_quantity"){
      updatePendingQuantity(data.quantity);
      add("agent",`Updated quantity to ${pendingQuantity}. Choose first, second, or third when you are ready.`);
      return;
    }
    if(data.action === "reorder_last"){
      const quantity = data.quantity || 1;
      if(!lastOrder || !lastOrder.product_id){ add("agent","I do not have a previous order to repeat yet."); return; }
      await orderProduct(lastOrder.product_id, quantity, lastOrder.shop_name || lastOrder.shop_id || "your last shop", false, lastOrder.wait_minutes, lastOrder.product_name);
      return;
    }
    if(data.action === "unsupported"){
      add("error", data.message || "Sorry, this demo can only order coffee right now. That product is not available yet.");
      return;
    }
    add("agent", data.message || "No problem. Tell me what coffee you want when you are ready.");
  }catch(e){
    thinking.remove();
    add("error",e.message);
  }
}
async function order(i){
  const x=latest[i]; if(!x) return;
  await orderProduct(x.product_id, pendingQuantity, x.shop_name, true, x.wait_minutes, x.product_name);
}
function estimatedPickupLabel(waitMinutes){
  const minutes = Math.max(3, Math.min(Number(waitMinutes || 8), 30));
  return new Date(Date.now() + minutes * 60000).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
}
async function orderProduct(productId, quantity, shopLabel, clearCards, waitMinutes, productLabel){
  if(orderInFlight){
    add("agent","I am already placing an order. Please wait for the result.");
    return;
  }
  orderInFlight = true;
  add("agent",`Creating order at <strong>${shopLabel}</strong>...`);
  try{
    if(!hasPaymentIdentity()){ add("error","Please register or login with an AigenticPay buyer API key first."); renderAuthState(); return; }
    const owner = apAccount.email || "u_001";
    const o=await api("/api/orders",{method:"POST",body:JSON.stringify({user_id:owner,product_id:productId,quantity:quantity,idempotency_key:`${owner}-${productId}-${quantity}-${Date.now()}`,buyer_email:apAccount.email,buyer_api_key:apAccount.api_key})});
    const pickupTime = estimatedPickupLabel(waitMinutes);
    const itemLabel = productLabel || o.product_id;
    add("agent",`Order confirmed: <strong>${o.order_id}</strong><br>Item: <strong>${itemLabel}</strong> x ${o.quantity}<br>Total $${Number(o.total).toFixed(2)}<br>Pick up at <strong>${shopLabel}</strong><br>Estimated pickup: <strong>${pickupTime}</strong><br>Payment ${o.payment_status}<br>Approval ${o.approval_id||"-"}<br>Tx ${o.tx_hash}${o.explorer_url?`<br><a target="_blank" href="${o.explorer_url}">Explorer</a>`:""}`);
    lastOrder = {product_id:o.product_id, product_name:itemLabel, shop_id:o.shop_id, shop_name:shopLabel, quantity:o.quantity, wait_minutes:waitMinutes};
    if(clearCards){
      pendingQuantity = 1; activeCardsMessage = null; document.querySelectorAll(".cards").forEach(c=>c.closest(".msg").remove());
    }
    loadOrders();
  }catch(e){ add("error",e.message); }
  finally{ orderInFlight = false; }
}
async function loadOrders(){
  try{
    if(!hasPaymentIdentity()){
      document.getElementById("orders").innerHTML="<span class='muted'>Sign in to see orders.</span>";
      return;
    }
    const rows=await api(`/api/orders?user_id=${encodeURIComponent(apAccount.email)}`);
    document.getElementById("orders").innerHTML=rows.map(o=>`<div class="order-card"><strong>${o.status}</strong><br>${o.order_id}<br>${o.shop_id}<br>$${Number(o.total).toFixed(2)}<br>${o.payment_status}<br>Approval ${o.approval_id||"-"}<br>${o.tx_hash}</div>`).join("") || "<span class='muted'>No orders yet.</span>";
  }catch(e){ document.getElementById("orders").innerHTML=e.message; }
}
document.getElementById("send").onclick=async()=>{
  const m=document.getElementById("message").value.trim(); if(!m) return;
  document.getElementById("message").value="";
  handleUserMessage(m);
};
document.getElementById("voice").onclick=async()=>{
  const button = document.getElementById("voice");
  if(recording && recorder){ recorder.stop(); return; }
  if(!navigator.mediaDevices || !window.MediaRecorder){ add("error","Voice recording is not supported in this browser."); return; }
  try{
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    chunks = [];
    const options = MediaRecorder.isTypeSupported("audio/webm") ? {mimeType:"audio/webm"} : {};
    recorder = new MediaRecorder(stream, options);
    recorder.ondataavailable = event => { if(event.data && event.data.size) chunks.push(event.data); };
    recorder.onstop = async()=>{
      recording = false;
      button.textContent = "Voice";
      button.classList.remove("recording");
      stream.getTracks().forEach(track => track.stop());
      const blob = new Blob(chunks, {type: recorder.mimeType || "audio/webm"});
      const form = new FormData();
      form.append("audio", blob, "voice.webm");
      add("agent","Transcribing voice...");
      try{
        const data = await apiForm("/api/voice/transcribe", form);
        await handleUserMessage(data.transcript, `Voice heard: ${data.transcript}`);
      }catch(e){
        if(e.transcript) add("user",`Voice heard: ${e.transcript}`);
        add("error",e.message);
      }
    };
    recorder.start();
    recording = true;
    button.textContent = "Stop";
    button.classList.add("recording");
    add("agent","Listening...");
    setTimeout(()=>{ if(recording && recorder) recorder.stop(); }, 15000);
  }catch(e){ add("error","Microphone permission was not granted."); }
};
renderAuthState();
</script>
</body>
</html>
        """
    )


@app.get("/merchant", response_class=HTMLResponse)
def merchant() -> HTMLResponse:
    if os.environ.get("ENABLE_MERCHANT_CONSOLE", "0") != "1":
        raise HTTPException(status_code=404, detail="Not found")

    shops = list_shops()
    buttons = "".join(
        f"<button onclick=\"selectShop('{escape(shop['id'])}')\">{escape(shop['name'])}</button>"
        for shop in shops
    )
    return HTMLResponse(
        f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Merchant Console</title>
  <style>
    body {{ margin:0; background:#f6f3ee; color:#1f2421; font-family:Arial,sans-serif; }}
    header {{ background:#20352d; color:white; padding:16px 22px; }}
    main {{ max-width:1120px; margin:0 auto; padding:20px; display:grid; grid-template-columns:240px 1fr; gap:16px; }}
    section {{ background:white; border:1px solid #ded8cf; border-radius:8px; padding:16px; }}
    button {{ margin:0 0 8px; border:0; border-radius:6px; padding:9px 11px; background:#176b54; color:white; cursor:pointer; }}
    .shops button {{ display:block; width:100%; text-align:left; }}
    .shops button.active {{ outline:2px solid #20352d; background:#0f5e49; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th,td {{ border-bottom:1px solid #eee7df; text-align:left; padding:9px; }}
    input {{ width:80px; padding:7px; }}
  </style>
</head>
<body>
<header><h1>Merchant Console</h1></header>
<main>
  <section class="shops"><h2>Shops</h2>{buttons}</section>
  <section><h2>Products</h2><div id="products"></div><h2>Orders</h2><div id="orders"></div></section>
</main>
<script>
let current = "{escape(shops[0]['id']) if shops else ''}";
async function api(path, options={{}}){{ const r=await fetch(path,{{headers:{{"Content-Type":"application/json"}},...options}}); const t=await r.text(); const d=t?JSON.parse(t):{{}}; if(!r.ok) throw new Error(d.detail||t); return d; }}
async function selectShop(id){{
 current=id;
 document.querySelectorAll(".shops button").forEach(btn => btn.classList.toggle("active", btn.getAttribute("onclick").includes(id)));
 document.getElementById("products").innerHTML="Loading products...";
 document.getElementById("orders").innerHTML="Loading orders...";
 await Promise.all([loadProducts(),loadOrders()]);
}}
async function loadProducts(){{
 const rows=await api(`/api/merchant/shops/${{current}}/products`);
 document.getElementById("products").innerHTML=`<table><tr><th>Name</th><th>Price</th><th>Inventory</th><th></th></tr>${{rows.map(r=>`<tr><td>${{r.name}}</td><td>$${{Number(r.price).toFixed(2)}}</td><td><input id="inv-${{r.id}}" value="${{r.inventory}}" type="number"></td><td><button onclick="saveInv('${{r.id}}')">Save</button></td></tr>`).join("")}}</table>`;
}}
async function loadOrders(){{
 const rows=await api(`/api/merchant/shops/${{current}}/orders`);
 document.getElementById("orders").innerHTML=rows.length
  ? `<table><tr><th>Order</th><th>Shop</th><th>Product</th><th>Total</th><th>Payment</th><th>Approval</th></tr>${{rows.map(r=>`<tr><td>${{r.id}}</td><td>${{r.shop_name}}</td><td>${{r.product_name}}</td><td>$${{Number(r.total).toFixed(2)}}</td><td>${{r.payment_status}}</td><td>${{r.approval_id||"-"}}</td></tr>`).join("")}}</table>`
  : "<p>No orders for this shop yet.</p>";
}}
async function saveInv(id){{ await api(`/api/merchant/products/${{id}}/inventory`,{{method:"POST",body:JSON.stringify({{inventory:Number(document.getElementById(`inv-${{id}}`).value)}})}}); await loadProducts(); }}
selectShop(current);
</script>
</body>
</html>
        """
    )
