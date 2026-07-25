# Coffee Agent Fast Render

Coffee Agent lets a user chat or speak with an agent to buy coffee. This version uses the AigenticPay external APIs from the PDF manual.

## AigenticPay APIs

The app connects to:

```text
POST /api/ext_register
POST /api/ext_login
POST /api/a2a_verify
```

Base URL:

```text
https://aigenticpay.onrender.com
```

Flow:

1. Register calls `/api/ext_register` and stores the returned buyer `api_key` in browser localStorage.
2. Login calls `/api/ext_login`. The current login API returns only true/false, so login users must paste a previously saved buyer API key.
3. Order confirmation calls `/api/a2a_verify` with the buyer API key in `X-API-Key`.
4. If AigenticPay approves, Coffee Agent creates the order and deducts inventory.
5. If AigenticPay rejects because of balance, limit, whitelist, or category rules, Coffee Agent shows the reason and does not deduct inventory.

Coffee shop merchant settings:

```text
merchant_id=00001
mcc_code=5814
```

The PDF API requires `merchant_id` for payment verification. `mcc_code` is stored in Coffee Agent for merchant/category context.

## Local Run

```powershell
cd C:\Users\Administrator\Desktop\coffee-agent-shop-fast-render
$env:AIGENTIC_PAY_ENABLED="1"
$env:AIGENTIC_PAY_API_MODE="a2a_verify"
$env:AIGENTIC_PAY_BASE_URL="https://aigenticpay.onrender.com"
$env:GROQ_API_KEY="your_groq_key"
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Render Config

Build Command:

```text
pip install -r requirements.txt
```

Start Command:

```text
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

Environment Variables:

```text
DATABASE_URL=<Neon connection string>
CLOUD_LLM_PROVIDER=groq
GROQ_API_KEY=<Groq API key>
GROQ_MODEL=llama-3.1-8b-instant
GROQ_TRANSCRIPTION_MODEL=whisper-large-v3-turbo
GROQ_TRANSCRIPTION_LANGUAGE=en
MAX_VOICE_UPLOAD_BYTES=10485760
AIGENTIC_PAY_ENABLED=1
AIGENTIC_PAY_API_MODE=a2a_verify
AIGENTIC_PAY_BASE_URL=https://aigenticpay.onrender.com
```

## Pages

```text
/                  user coffee agent app
/health            health check
/api/llm/status    LLM and voice model status
```

The merchant console is hidden from users. `/merchant` returns 404 by default. To enable it for local admin testing, start with:

```powershell
$env:ENABLE_MERCHANT_CONSOLE="1"
```
