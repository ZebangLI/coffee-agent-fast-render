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
from .llm import llm_status, parse_intent, transcribe_audio
from .models import (
    ChatRequest,
    ChatResponse,
    CreateOrderRequest,
    Location,
    OrderResponse,
    UpdateInventoryRequest,
    VoiceChatResponse,
)
from .payment import record_payment

app = FastAPI(title="Coffee Agent Fast Render", version="0.2.0")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/llm/status")
def api_llm_status() -> dict:
    return llm_status()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return _chat_from_text(request.message, request.location)


@app.post("/api/voice/chat", response_model=VoiceChatResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    user_id: str = Form("u_001"),
    lat: float = Form(40.731),
    lng: float = Form(-73.992),
) -> VoiceChatResponse:
    del user_id
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

    chat_response = _chat_from_text(transcript, Location(lat=lat, lng=lng))
    return VoiceChatResponse(
        transcript=transcript,
        intent=chat_response.intent,
        recommendations=chat_response.recommendations,
    )


@app.get("/api/orders", response_model=list[OrderResponse])
def api_list_orders() -> list[OrderResponse]:
    return list_orders()


@app.post("/api/orders", response_model=OrderResponse)
def create_order(request: CreateOrderRequest) -> OrderResponse:
    existing = get_order_by_idempotency_key(request.idempotency_key)
    if existing:
        return existing

    product = get_product_for_order(request.product_id)
    if product["inventory"] < request.quantity:
        raise HTTPException(status_code=409, detail="Not enough inventory")

    total = round(product["price"] * request.quantity, 2)
    order_id = f"ord_{uuid4().hex[:10]}"
    try:
        payment = record_payment(
            {
                "source_app": "coffee-agent-fast-render",
                "external_order_id": order_id,
                "user_ref": request.user_id,
                "shop_id": product["shop_id"],
                "shop_name": product["shop_name"],
                "product_id": product["id"],
                "product_name": product["name"],
                "amount": total,
                "currency": "USD",
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
        raise HTTPException(status_code=503, detail="Cloud LLM did not detect a coffee order.")
    recommendations = recommend_products(intent.drink, location)
    return ChatResponse(intent=intent, recommendations=recommendations)


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
  </style>
</head>
<body>
<header><h1>Coffee Agent</h1><span>Groq LLM -> Fast MCP Hub -> Order</span></header>
<main>
  <section class="chat">
    <div class="head"><strong>Chat</strong><a href="/merchant" target="_blank">Merchant</a></div>
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
let recorder = null;
let chunks = [];
let recording = false;
const log = document.getElementById("log");
function add(role, html){ const n=document.createElement("div"); n.className=`msg ${role}`; n.innerHTML=html; log.appendChild(n); log.scrollTop=log.scrollHeight; return n; }
async function api(path, options={}){ const r=await fetch(path,{headers:{"Content-Type":"application/json"},...options}); const t=await r.text(); const d=t?JSON.parse(t):{}; if(!r.ok) throw new Error(d.detail||t||r.status); return d; }
async function apiForm(path, form){ const r=await fetch(path,{method:"POST",body:form}); const t=await r.text(); const d=t?JSON.parse(t):{}; if(!r.ok) throw new Error(d.detail||t||r.status); return d; }
function renderRecs(data){
  latest=data.recommendations||[];
  if(!latest.length){ add("agent","No matching coffee nearby."); return; }
  add("agent", `Intent: <strong>${data.intent.drink}</strong><div class="cards">${latest.map((x,i)=>`
    <div class="card"><div class="top"><strong>${i+1}. ${x.shop_name}</strong><strong>$${Number(x.price).toFixed(2)}</strong></div>
    <div class="muted">${x.product_name}</div><span class="pill">${x.distance_km} km</span><span class="pill">${x.wait_minutes} min</span><span class="pill">score ${x.score}</span>
    <div><button onclick="order(${i})">Order this</button></div></div>`).join("")}</div>`);
}
async function sendText(message){
  add("user",message); add("agent","Checking nearby coffee shops...");
  try{ renderRecs(await api("/api/chat",{method:"POST",body:JSON.stringify({message})})); }catch(e){ add("error",e.message); }
}
async function order(i){
  const x=latest[i]; if(!x) return;
  add("agent",`Creating order at <strong>${x.shop_name}</strong>...`);
  try{
    const o=await api("/api/orders",{method:"POST",body:JSON.stringify({product_id:x.product_id,quantity:1,idempotency_key:`u_001-${x.product_id}-${Date.now()}`})});
    add("agent",`Order confirmed: <strong>${o.order_id}</strong><br>Total $${Number(o.total).toFixed(2)}<br>Payment ${o.payment_status}<br>Tx ${o.tx_hash}${o.explorer_url?`<br><a target="_blank" href="${o.explorer_url}">Explorer</a>`:""}`);
    latest=[]; document.querySelectorAll(".cards").forEach(c=>c.closest(".msg").remove()); loadOrders();
  }catch(e){ add("error",e.message); }
}
async function loadOrders(){
  try{
    const rows=await api("/api/orders");
    document.getElementById("orders").innerHTML=rows.map(o=>`<div class="order-card"><strong>${o.status}</strong><br>${o.order_id}<br>${o.shop_id}<br>$${Number(o.total).toFixed(2)}<br>${o.payment_status}<br>${o.tx_hash}</div>`).join("") || "<span class='muted'>No orders yet.</span>";
  }catch(e){ document.getElementById("orders").innerHTML=e.message; }
}
document.getElementById("send").onclick=async()=>{
  const m=document.getElementById("message").value.trim(); if(!m) return;
  document.getElementById("message").value="";
  sendText(m);
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
        const data = await apiForm("/api/voice/chat", form);
        add("user",`Voice: ${data.transcript}`);
        renderRecs(data);
      }catch(e){ add("error",e.message); }
    };
    recorder.start();
    recording = true;
    button.textContent = "Stop";
    button.classList.add("recording");
    add("agent","Listening...");
    setTimeout(()=>{ if(recording && recorder) recorder.stop(); }, 15000);
  }catch(e){ add("error","Microphone permission was not granted."); }
};
loadOrders();
</script>
</body>
</html>
        """
    )


@app.get("/merchant", response_class=HTMLResponse)
def merchant() -> HTMLResponse:
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
  ? `<table><tr><th>Order</th><th>Shop</th><th>Product</th><th>Total</th><th>Status</th></tr>${{rows.map(r=>`<tr><td>${{r.id}}</td><td>${{r.shop_name}}</td><td>${{r.product_name}}</td><td>$${{Number(r.total).toFixed(2)}}</td><td>${{r.status}}</td></tr>`).join("")}}</table>`
  : "<p>No orders for this shop yet.</p>";
}}
async function saveInv(id){{ await api(`/api/merchant/products/${{id}}/inventory`,{{method:"POST",body:JSON.stringify({{inventory:Number(document.getElementById(`inv-${{id}}`).value)}})}}); await loadProducts(); }}
selectShop(current);
</script>
</body>
</html>
        """
    )
