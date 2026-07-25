# Coffee Agent Fast Render

这是 Coffee Agent 的线上部署快版本，目标是让 Render demo 响应更快、结构更简单。

## 当前功能

- 用户可以在网页聊天里输入咖啡需求。
- 用户也可以点击 `Voice` 录音，用 Groq Whisper 转成文字后继续下单流程。
- Groq LLM 负责从用户话里提取咖啡意图。
- 如果已经有推荐列表，用户可以打字或语音说 `second one`，LLM 会识别为选择第二个选项并直接下单。
- 后端根据位置、等待时间、价格和库存推荐三家咖啡店。
- 用户确认后先调用 AigenticPay `a2a_verify` 做风控校验，成功后创建订单并扣减库存。
- 如果 AigenticPay 因余额、每日限额、类别限制、白名单等规则拒绝交易，页面会直接提示拒绝原因，不扣库存。
- 商家后台可以按店铺查看商品、库存和订单。
- 数据库使用一个 Neon/Postgres 数据库，核心表是 `shops`、`products`、`orders`。

## 简化点

- 不再用 stdio MCP 子进程，避免 Render 上每次请求启动多个 Python 子进程。
- 保留 MCP 思路，但实现成轻量的 in-process Fast MCP Hub / shop tools。
- 本地没有 `DATABASE_URL` 时使用 SQLite。
- Render 有 `DATABASE_URL` 时使用 Neon Postgres。

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
GROQ_TRANSCRIPTION_MODEL=whisper-large-v3-turbo
GROQ_TRANSCRIPTION_LANGUAGE=en
MAX_VOICE_UPLOAD_BYTES=10485760
AIGENTIC_PAY_ENABLED=1
AIGENTIC_PAY_API_MODE=a2a_verify
AIGENTIC_PAY_BASE_URL=https://aigenticpay.onrender.com
AIGENTIC_PAY_API_KEY=<AigenticPay API key>
AIGENTIC_PAY_USER_EMAIL=lizebang2017@icloud.com
```

Coffee shop demo uses:

```text
merchant_id=00001
mcc_code=5814
```

The PDF API requires `merchant_id` for AigenticPay rule checking. `mcc_code` is stored in our database for merchant/category context; the actual rule decision is made by AigenticPay inside `/api/a2a_verify`.

## Pages

```text
/                  User chat app
/merchant          Merchant console
/health            Health check
/api/llm/status    LLM and voice model status
```

## 本地运行

```powershell
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

本地语音录音需要浏览器允许麦克风权限。线上 Render 是 HTTPS，浏览器麦克风权限可以正常使用。
