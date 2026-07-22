# Coffee Agent Fast Render

这是 Coffee Agent 的线上部署快版本，目标是让 Render demo 响应更快、结构更简单。

## 简化点

- 不再使用 stdio MCP 子进程，避免 Render 上每次请求启动多个 Python 子进程。
- 保留 MCP 思路，但实现为轻量的 in-process Fast MCP Hub / shop tools。
- 数据库压缩为一个数据库三张核心表：
  - `shops`
  - `products`
  - `orders`
- 本地没有 `DATABASE_URL` 时使用 SQLite。
- Render 有 `DATABASE_URL` 时使用 Neon Postgres。
- 云端 LLM 使用 Groq。

## Render 配置

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
AIGENTIC_PAY_ENABLED=0
AIGENTIC_PAY_BASE_URL=https://aigenticpay.onrender.com
AIGENTIC_PAY_API_KEY=<AigenticPay API key>
AIGENTIC_PAY_USER_EMAIL=lizebang2017@icloud.com
AIGENTIC_PAY_USE_VIRTUAL_CARD=0
AIGENTIC_PAY_REQUIRE_ONCHAIN=0
```

## Pages

```text
/          User chat app
/merchant  Merchant console
/health    Health check
/api/llm/status
```

## 本地运行

```powershell
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

