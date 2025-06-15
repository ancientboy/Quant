# API设计文档

## 概述

本文档定义了量化交易系统中各个微服务的API接口规范，包括行情数据服务器和选币系统的REST API和WebSocket接口。

## 通用规范

### 1. 基础信息

- **协议**: HTTP/1.1, WebSocket
- **数据格式**: JSON
- **字符编码**: UTF-8
- **时间格式**: Unix时间戳（毫秒）
- **API版本**: v1

### 2. 认证方式

```http
Authorization: Bearer <JWT_TOKEN>
X-API-Key: <API_KEY>
```

### 3. 通用响应格式

#### 成功响应
```json
{
  "success": true,
  "data": {},
  "message": "操作成功",
  "timestamp": 1703123456789,
  "request_id": "req_123456789"
}
```

#### 错误响应
```json
{
  "success": false,
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "参数无效",
    "details": "symbol参数不能为空"
  },
  "timestamp": 1703123456789,
  "request_id": "req_123456789"
}
```

### 4. 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 201 | 创建成功 |
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 403 | 禁止访问 |
| 404 | 资源不存在 |
| 429 | 请求频率限制 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 |

### 5. 错误代码

| 错误代码 | 说明 |
|----------|------|
| INVALID_PARAMETER | 参数无效 |
| MISSING_PARAMETER | 缺少必需参数 |
| UNAUTHORIZED | 未授权访问 |
| RATE_LIMIT_EXCEEDED | 超出频率限制 |
| EXCHANGE_ERROR | 交易所接口错误 |
| DATA_NOT_FOUND | 数据不存在 |
| INTERNAL_ERROR | 内部服务错误 |

---

## 行情数据服务器 API

### 基础URL
```
开发环境: http://localhost:8001/api/v1
生产环境: https://api.quant-system.com/market-data/v1
```

### 1. 实时行情接口

#### 1.1 获取单个币种行情

```http
GET /ticker/{symbol}
```

**路径参数:**
- `symbol` (string, required): 交易对符号，如 "BTC/USDT"

**查询参数:**
- `exchange` (string, optional): 交易所名称，默认返回所有交易所数据

**响应示例:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTC/USDT",
    "exchange": "binance",
    "price": 43250.50,
    "volume_24h": 1234567.89,
    "change_24h": 2.35,
    "change_24h_percent": 5.76,
    "high_24h": 43500.00,
    "low_24h": 42000.00,
    "bid": 43249.50,
    "ask": 43251.00,
    "bid_volume": 0.5,
    "ask_volume": 0.3,
    "timestamp": 1703123456789
  }
}
```

#### 1.2 获取多个币种行情

```http
GET /tickers
```

**查询参数:**
- `symbols` (string, optional): 逗号分隔的交易对列表，如 "BTC/USDT,ETH/USDT"
- `exchange` (string, optional): 交易所名称
- `limit` (integer, optional): 返回数量限制，默认100，最大1000
- `sort` (string, optional): 排序字段，可选值: price, volume, change
- `order` (string, optional): 排序方向，asc/desc，默认desc

**响应示例:**
```json
{
  "success": true,
  "data": {
    "tickers": [
      {
        "symbol": "BTC/USDT",
        "exchange": "binance",
        "price": 43250.50,
        "volume_24h": 1234567.89,
        "change_24h_percent": 5.76,
        "timestamp": 1703123456789
      }
    ],
    "total": 150,
    "page": 1,
    "limit": 100
  }
}
```

### 2. K线数据接口

#### 2.1 获取K线数据

```http
GET /klines/{symbol}
```

**路径参数:**
- `symbol` (string, required): 交易对符号

**查询参数:**
- `timeframe` (string, required): 时间周期，支持: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w
- `limit` (integer, optional): 返回数量，默认100，最大1000
- `start` (integer, optional): 开始时间戳（毫秒）
- `end` (integer, optional): 结束时间戳（毫秒）
- `exchange` (string, optional): 交易所名称

**响应示例:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "klines": [
      {
        "timestamp": 1703120000000,
        "open": 43000.00,
        "high": 43300.00,
        "low": 42900.00,
        "close": 43250.50,
        "volume": 123.45,
        "trades": 1250
      }
    ],
    "count": 100
  }
}
```

### 3. 订单簿接口

#### 3.1 获取订单簿

```http
GET /orderbook/{symbol}
```

**路径参数:**
- `symbol` (string, required): 交易对符号

**查询参数:**
- `limit` (integer, optional): 深度限制，默认20，最大100
- `exchange` (string, optional): 交易所名称

**响应示例:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTC/USDT",
    "exchange": "binance",
    "bids": [
      [43249.50, 0.5],
      [43249.00, 1.2]
    ],
    "asks": [
      [43251.00, 0.3],
      [43251.50, 0.8]
    ],
    "timestamp": 1703123456789
  }
}
```

### 4. 交易所信息接口

#### 4.1 获取支持的交易所列表

```http
GET /exchanges
```

**响应示例:**
```json
{
  "success": true,
  "data": {
    "exchanges": [
      {
        "id": "binance",
        "name": "Binance",
        "status": "active",
        "rate_limit": 1200,
        "supported_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        "features": {
          "ticker": true,
          "klines": true,
          "orderbook": true,
          "websocket": true
        }
      }
    ]
  }
}
```

#### 4.2 获取交易对列表

```http
GET /symbols
```

**查询参数:**
- `exchange` (string, optional): 交易所名称
- `base` (string, optional): 基础货币，如 "BTC"
- `quote` (string, optional): 计价货币，如 "USDT"
- `active` (boolean, optional): 是否只返回活跃交易对

**响应示例:**
```json
{
  "success": true,
  "data": {
    "symbols": [
      {
        "symbol": "BTC/USDT",
        "base": "BTC",
        "quote": "USDT",
        "active": true,
        "exchange": "binance",
        "min_amount": 0.001,
        "max_amount": 1000,
        "price_precision": 2,
        "amount_precision": 6
      }
    ],
    "total": 500
  }
}
```

### 5. WebSocket接口

#### 5.1 实时行情推送

**连接URL:**
```
ws://localhost:8001/ws/ticker/{symbol}
```

**订阅消息:**
```json
{
  "action": "subscribe",
  "channel": "ticker",
  "symbol": "BTC/USDT",
  "exchange": "binance"
}
```

**推送消息:**
```json
{
  "channel": "ticker",
  "symbol": "BTC/USDT",
  "data": {
    "price": 43250.50,
    "volume_24h": 1234567.89,
    "change_24h_percent": 5.76,
    "timestamp": 1703123456789
  }
}
```

#### 5.2 K线数据推送

**连接URL:**
```
ws://localhost:8001/ws/klines/{symbol}
```

**订阅消息:**
```json
{
  "action": "subscribe",
  "channel": "klines",
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "exchange": "binance"
}
```

**推送消息:**
```json
{
  "channel": "klines",
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "data": {
    "timestamp": 1703120000000,
    "open": 43000.00,
    "high": 43300.00,
    "low": 42900.00,
    "close": 43250.50,
    "volume": 123.45,
    "is_closed": false
  }
}
```

---

## 选币系统 API

### 基础URL
```
开发环境: http://localhost:8002/api/v1
生产环境: https://api.quant-system.com/coin-selection/v1
```

### 1. 策略管理接口

#### 1.1 获取策略列表

```http
GET /strategies
```

**查询参数:**
- `type` (string, optional): 策略类型，technical/fundamental/ml
- `enabled` (boolean, optional): 是否启用
- `user_id` (integer, optional): 用户ID

**响应示例:**
```json
{
  "success": true,
  "data": {
    "strategies": [
      {
        "id": 1,
        "name": "RSI策略",
        "type": "technical",
        "description": "基于RSI指标的选币策略",
        "parameters": {
          "oversold_threshold": 30,
          "overbought_threshold": 70,
          "period": 14
        },
        "weight": 0.3,
        "enabled": true,
        "created_at": 1703123456789,
        "updated_at": 1703123456789
      }
    ],
    "total": 10
  }
}
```

#### 1.2 创建策略

```http
POST /strategies
```

**请求体:**
```json
{
  "name": "自定义RSI策略",
  "type": "technical",
  "description": "我的RSI策略",
  "parameters": {
    "oversold_threshold": 25,
    "overbought_threshold": 75,
    "period": 14
  },
  "weight": 0.4,
  "enabled": true
}
```

#### 1.3 更新策略

```http
PUT /strategies/{strategy_id}
```

#### 1.4 删除策略

```http
DELETE /strategies/{strategy_id}
```

### 2. 选币结果接口

#### 2.1 获取最新选币结果

```http
GET /selections/latest
```

**查询参数:**
- `limit` (integer, optional): 返回数量，默认50
- `min_score` (float, optional): 最小评分
- `exchanges` (string, optional): 交易所列表，逗号分隔

**响应示例:**
```json
{
  "success": true,
  "data": {
    "selections": [
      {
        "rank": 1,
        "symbol": "BTC/USDT",
        "exchange": "binance",
        "total_score": 85.6,
        "strategy_scores": {
          "rsi_strategy": 90.0,
          "volume_strategy": 80.0,
          "macd_strategy": 87.5
        },
        "current_price": 43250.50,
        "change_24h_percent": 5.76,
        "volume_24h": 1234567.89,
        "market_cap": 850000000000,
        "timestamp": 1703123456789
      }
    ],
    "metadata": {
      "total_analyzed": 500,
      "strategies_used": ["rsi_strategy", "volume_strategy", "macd_strategy"],
      "analysis_time": 1703123456789,
      "next_analysis": 1703127056789
    }
  }
}
```

#### 2.2 获取历史选币结果

```http
GET /selections/history
```

**查询参数:**
- `start` (integer, required): 开始时间戳
- `end` (integer, required): 结束时间戳
- `symbol` (string, optional): 特定交易对
- `page` (integer, optional): 页码，默认1
- `limit` (integer, optional): 每页数量，默认20

### 3. 回测接口

#### 3.1 创建回测任务

```http
POST /backtest
```

**请求体:**
```json
{
  "name": "RSI策略回测",
  "strategies": [1, 2, 3],
  "start_date": "2023-01-01",
  "end_date": "2023-12-01",
  "initial_capital": 10000,
  "rebalance_frequency": "weekly",
  "max_positions": 10,
  "parameters": {
    "commission": 0.001,
    "slippage": 0.0005
  }
}
```

**响应示例:**
```json
{
  "success": true,
  "data": {
    "task_id": "bt_123456789",
    "status": "pending",
    "estimated_time": 300,
    "created_at": 1703123456789
  }
}
```

#### 3.2 获取回测状态

```http
GET /backtest/{task_id}/status
```

**响应示例:**
```json
{
  "success": true,
  "data": {
    "task_id": "bt_123456789",
    "status": "running",
    "progress": 65.5,
    "estimated_remaining": 120,
    "current_step": "计算技术指标"
  }
}
```

#### 3.3 获取回测结果

```http
GET /backtest/{task_id}/result
```

**响应示例:**
```json
{
  "success": true,
  "data": {
    "task_id": "bt_123456789",
    "status": "completed",
    "performance": {
      "total_return": 0.156,
      "annual_return": 0.142,
      "sharpe_ratio": 1.85,
      "max_drawdown": 0.085,
      "win_rate": 0.68,
      "profit_factor": 2.34
    },
    "trades": [
      {
        "symbol": "BTC/USDT",
        "side": "buy",
        "quantity": 0.1,
        "price": 40000.00,
        "timestamp": 1703123456789,
        "pnl": 325.50
      }
    ],
    "equity_curve": [
      {
        "timestamp": 1703123456789,
        "equity": 10325.50,
        "drawdown": 0.025
      }
    ]
  }
}
```

### 4. 分析接口

#### 4.1 获取技术指标

```http
GET /analysis/indicators/{symbol}
```

**查询参数:**
- `indicators` (string, required): 指标列表，逗号分隔，如 "rsi,macd,bb"
- `timeframe` (string, required): 时间周期
- `period` (integer, optional): 计算周期

**响应示例:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "indicators": {
      "rsi": {
        "value": 65.5,
        "signal": "neutral",
        "timestamp": 1703123456789
      },
      "macd": {
        "macd": 125.5,
        "signal": 118.2,
        "histogram": 7.3,
        "signal": "bullish",
        "timestamp": 1703123456789
      }
    }
  }
}
```

#### 4.2 获取市场概览

```http
GET /analysis/market-overview
```

**响应示例:**
```json
{
  "success": true,
  "data": {
    "market_sentiment": "bullish",
    "total_market_cap": 1750000000000,
    "btc_dominance": 0.52,
    "fear_greed_index": 75,
    "top_gainers": [
      {
        "symbol": "ETH/USDT",
        "change_24h_percent": 8.5,
        "price": 2450.00
      }
    ],
    "top_losers": [
      {
        "symbol": "ADA/USDT",
        "change_24h_percent": -5.2,
        "price": 0.485
      }
    ],
    "volume_leaders": [
      {
        "symbol": "BTC/USDT",
        "volume_24h": 1234567890,
        "volume_change_percent": 15.6
      }
    ]
  }
}
```

### 5. 配置接口

#### 5.1 获取系统配置

```http
GET /config
```

#### 5.2 更新系统配置

```http
PUT /config
```

**请求体:**
```json
{
  "selection": {
    "max_coins": 50,
    "min_score": 60,
    "rebalance_interval": 3600
  },
  "filters": {
    "min_market_cap": 100000000,
    "min_daily_volume": 5000000,
    "blacklist": ["USDT", "BUSD"],
    "whitelist_exchanges": ["binance", "okx"]
  }
}
```

---

## WebSocket通用协议

### 1. 连接认证

```json
{
  "action": "auth",
  "token": "your_jwt_token"
}
```

### 2. 心跳机制

**客户端发送:**
```json
{
  "action": "ping"
}
```

**服务端响应:**
```json
{
  "action": "pong",
  "timestamp": 1703123456789
}
```

### 3. 错误处理

```json
{
  "action": "error",
  "error": {
    "code": "SUBSCRIPTION_FAILED",
    "message": "订阅失败",
    "details": "无效的交易对"
  }
}
```

### 4. 订阅管理

**订阅:**
```json
{
  "action": "subscribe",
  "channel": "ticker",
  "params": {
    "symbol": "BTC/USDT",
    "exchange": "binance"
  }
}
```

**取消订阅:**
```json
{
  "action": "unsubscribe",
  "channel": "ticker",
  "params": {
    "symbol": "BTC/USDT"
  }
}
```

---

## 限流规则

### 1. REST API限流

| 端点类型 | 限制 | 时间窗口 |
|----------|------|----------|
| 行情数据 | 1000次/分钟 | 60秒 |
| 选币结果 | 100次/分钟 | 60秒 |
| 策略管理 | 50次/分钟 | 60秒 |
| 回测任务 | 10次/小时 | 3600秒 |

### 2. WebSocket限流

- 每个连接最多订阅50个频道
- 每秒最多发送10条消息
- 连接空闲超过5分钟自动断开

---

## SDK示例

### Python SDK

```python
import asyncio
from quant_api import MarketDataClient, CoinSelectionClient

# 初始化客户端
market_client = MarketDataClient(
    base_url="http://localhost:8001/api/v1",
    api_key="your_api_key"
)

selection_client = CoinSelectionClient(
    base_url="http://localhost:8002/api/v1",
    api_key="your_api_key"
)

async def main():
    # 获取行情数据
    ticker = await market_client.get_ticker("BTC/USDT")
    print(f"BTC价格: {ticker['price']}")
    
    # 获取选币结果
    selections = await selection_client.get_latest_selections(limit=10)
    for coin in selections['selections']:
        print(f"{coin['symbol']}: {coin['total_score']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript SDK

```javascript
import { MarketDataClient, CoinSelectionClient } from 'quant-api-js';

const marketClient = new MarketDataClient({
  baseURL: 'http://localhost:8001/api/v1',
  apiKey: 'your_api_key'
});

const selectionClient = new CoinSelectionClient({
  baseURL: 'http://localhost:8002/api/v1',
  apiKey: 'your_api_key'
});

// 获取行情数据
const ticker = await marketClient.getTicker('BTC/USDT');
console.log(`BTC价格: ${ticker.data.price}`);

// WebSocket连接
const ws = marketClient.createWebSocket();
ws.subscribe('ticker', { symbol: 'BTC/USDT' });
ws.on('ticker', (data) => {
  console.log('实时价格:', data.price);
});
```

---

*本API文档将随着系统开发进度持续更新和完善。*