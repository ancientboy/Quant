# 数据库设计文档

## 概述

本文档详细描述了量化交易系统的数据库设计，包括PostgreSQL关系型数据库、InfluxDB时序数据库和Redis缓存数据库的表结构设计。

## 数据库架构

### 1. 数据库分工

| 数据库类型 | 用途 | 存储内容 |
|------------|------|----------|
| PostgreSQL | 关系型数据 | 用户信息、策略配置、系统配置、交易记录 |
| InfluxDB | 时序数据 | 行情数据、K线数据、技术指标、系统监控 |
| Redis | 缓存数据 | 实时行情、会话信息、计算结果、消息队列 |

### 2. 数据流向

```
交易所API → 行情服务器 → InfluxDB (历史数据)
                    ↓
                  Redis (实时缓存)
                    ↓
选币系统 → PostgreSQL (结果存储) → Web前端
```

---

## PostgreSQL 数据库设计

### 1. 用户管理模块

#### 1.1 用户表 (users)

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    salt VARCHAR(32) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    avatar_url VARCHAR(255),
    phone VARCHAR(20),
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'zh-CN',
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('admin', 'user', 'viewer')),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended')),
    email_verified BOOLEAN DEFAULT FALSE,
    phone_verified BOOLEAN DEFAULT FALSE,
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    two_factor_secret VARCHAR(32),
    last_login_at TIMESTAMP,
    last_login_ip INET,
    login_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
);

-- 索引
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_created_at ON users(created_at);
```

#### 1.2 用户会话表 (user_sessions)

```sql
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    refresh_token_hash VARCHAR(255),
    device_info JSONB,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token_hash ON user_sessions(token_hash);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);
```

### 2. 交易所管理模块

#### 2.1 交易所配置表 (exchanges)

```sql
CREATE TABLE exchanges (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    api_endpoint VARCHAR(255) NOT NULL,
    websocket_endpoint VARCHAR(255),
    sandbox_endpoint VARCHAR(255),
    rate_limit_per_minute INTEGER DEFAULT 1000,
    rate_limit_per_second INTEGER DEFAULT 10,
    supported_timeframes TEXT[] DEFAULT ARRAY['1m','5m','15m','30m','1h','4h','1d'],
    features JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance')),
    priority INTEGER DEFAULT 100,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_exchanges_name ON exchanges(name);
CREATE INDEX idx_exchanges_status ON exchanges(status);
```

#### 2.2 交易对表 (trading_pairs)

```sql
CREATE TABLE trading_pairs (
    id SERIAL PRIMARY KEY,
    exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
    symbol VARCHAR(20) NOT NULL,
    base_currency VARCHAR(10) NOT NULL,
    quote_currency VARCHAR(10) NOT NULL,
    min_amount DECIMAL(20,8),
    max_amount DECIMAL(20,8),
    min_price DECIMAL(20,8),
    max_price DECIMAL(20,8),
    price_precision INTEGER DEFAULT 8,
    amount_precision INTEGER DEFAULT 8,
    maker_fee DECIMAL(6,4) DEFAULT 0.001,
    taker_fee DECIMAL(6,4) DEFAULT 0.001,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(exchange_id, symbol)
);

-- 索引
CREATE INDEX idx_trading_pairs_symbol ON trading_pairs(symbol);
CREATE INDEX idx_trading_pairs_exchange_id ON trading_pairs(exchange_id);
CREATE INDEX idx_trading_pairs_base_currency ON trading_pairs(base_currency);
CREATE INDEX idx_trading_pairs_quote_currency ON trading_pairs(quote_currency);
CREATE INDEX idx_trading_pairs_is_active ON trading_pairs(is_active);
```

### 3. 策略管理模块

#### 3.1 策略模板表 (strategy_templates)

```sql
CREATE TABLE strategy_templates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('technical', 'fundamental', 'ml', 'hybrid')),
    category VARCHAR(50),
    description TEXT,
    algorithm_class VARCHAR(100) NOT NULL,
    default_parameters JSONB NOT NULL DEFAULT '{}',
    parameter_schema JSONB NOT NULL DEFAULT '{}',
    required_data_types TEXT[] DEFAULT ARRAY['klines'],
    min_data_points INTEGER DEFAULT 100,
    complexity_level INTEGER DEFAULT 1 CHECK (complexity_level BETWEEN 1 AND 5),
    is_public BOOLEAN DEFAULT TRUE,
    created_by INTEGER REFERENCES users(id),
    version VARCHAR(20) DEFAULT '1.0.0',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'deprecated', 'beta')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_strategy_templates_type ON strategy_templates(type);
CREATE INDEX idx_strategy_templates_category ON strategy_templates(category);
CREATE INDEX idx_strategy_templates_status ON strategy_templates(status);
```

#### 3.2 用户策略配置表 (user_strategies)

```sql
CREATE TABLE user_strategies (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    template_id INTEGER NOT NULL REFERENCES strategy_templates(id),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parameters JSONB NOT NULL DEFAULT '{}',
    weight DECIMAL(5,4) DEFAULT 1.0000 CHECK (weight >= 0 AND weight <= 1),
    is_enabled BOOLEAN DEFAULT TRUE,
    performance_metrics JSONB DEFAULT '{}',
    last_executed_at TIMESTAMP,
    execution_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, name)
);

-- 索引
CREATE INDEX idx_user_strategies_user_id ON user_strategies(user_id);
CREATE INDEX idx_user_strategies_template_id ON user_strategies(template_id);
CREATE INDEX idx_user_strategies_is_enabled ON user_strategies(is_enabled);
```

### 4. 选币结果模块

#### 4.1 选币任务表 (selection_tasks)

```sql
CREATE TABLE selection_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id),
    name VARCHAR(100) NOT NULL,
    strategy_ids INTEGER[] NOT NULL,
    filters JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress DECIMAL(5,2) DEFAULT 0.00,
    total_symbols INTEGER DEFAULT 0,
    processed_symbols INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    execution_time_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_selection_tasks_user_id ON selection_tasks(user_id);
CREATE INDEX idx_selection_tasks_status ON selection_tasks(status);
CREATE INDEX idx_selection_tasks_created_at ON selection_tasks(created_at);
```

#### 4.2 选币结果表 (coin_selections)

```sql
CREATE TABLE coin_selections (
    id BIGSERIAL PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES selection_tasks(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id),
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    rank INTEGER NOT NULL,
    total_score DECIMAL(6,3) NOT NULL,
    strategy_scores JSONB NOT NULL DEFAULT '{}',
    market_data JSONB DEFAULT '{}',
    analysis_data JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(task_id, symbol, exchange)
);

-- 索引
CREATE INDEX idx_coin_selections_task_id ON coin_selections(task_id);
CREATE INDEX idx_coin_selections_user_id ON coin_selections(user_id);
CREATE INDEX idx_coin_selections_symbol ON coin_selections(symbol);
CREATE INDEX idx_coin_selections_total_score ON coin_selections(total_score DESC);
CREATE INDEX idx_coin_selections_created_at ON coin_selections(created_at);
```

### 5. 回测模块

#### 5.1 回测任务表 (backtest_tasks)

```sql
CREATE TABLE backtest_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id),
    name VARCHAR(100) NOT NULL,
    strategy_ids INTEGER[] NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,2) NOT NULL,
    rebalance_frequency VARCHAR(20) DEFAULT 'weekly' CHECK (rebalance_frequency IN ('daily', 'weekly', 'monthly')),
    max_positions INTEGER DEFAULT 10,
    commission DECIMAL(6,4) DEFAULT 0.001,
    slippage DECIMAL(6,4) DEFAULT 0.0005,
    benchmark VARCHAR(20) DEFAULT 'BTC/USDT',
    parameters JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress DECIMAL(5,2) DEFAULT 0.00,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    execution_time_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_backtest_tasks_user_id ON backtest_tasks(user_id);
CREATE INDEX idx_backtest_tasks_status ON backtest_tasks(status);
CREATE INDEX idx_backtest_tasks_created_at ON backtest_tasks(created_at);
```

#### 5.2 回测结果表 (backtest_results)

```sql
CREATE TABLE backtest_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES backtest_tasks(id) ON DELETE CASCADE,
    performance_metrics JSONB NOT NULL DEFAULT '{}',
    risk_metrics JSONB NOT NULL DEFAULT '{}',
    trade_statistics JSONB NOT NULL DEFAULT '{}',
    equity_curve JSONB DEFAULT '{}',
    drawdown_curve JSONB DEFAULT '{}',
    monthly_returns JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_backtest_results_task_id ON backtest_results(task_id);
```

#### 5.3 回测交易记录表 (backtest_trades)

```sql
CREATE TABLE backtest_trades (
    id BIGSERIAL PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES backtest_tasks(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    commission DECIMAL(20,8) DEFAULT 0,
    slippage DECIMAL(20,8) DEFAULT 0,
    trade_value DECIMAL(20,8) NOT NULL,
    pnl DECIMAL(20,8),
    pnl_percent DECIMAL(8,4),
    portfolio_weight DECIMAL(6,4),
    reason TEXT,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_backtest_trades_task_id ON backtest_trades(task_id);
CREATE INDEX idx_backtest_trades_symbol ON backtest_trades(symbol);
CREATE INDEX idx_backtest_trades_timestamp ON backtest_trades(timestamp);
```

### 6. 系统配置模块

#### 6.1 系统配置表 (system_configs)

```sql
CREATE TABLE system_configs (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    key VARCHAR(100) NOT NULL,
    value JSONB NOT NULL,
    data_type VARCHAR(20) DEFAULT 'json' CHECK (data_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    description TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    is_editable BOOLEAN DEFAULT TRUE,
    validation_rules JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by INTEGER REFERENCES users(id),
    
    UNIQUE(category, key)
);

-- 索引
CREATE INDEX idx_system_configs_category ON system_configs(category);
CREATE INDEX idx_system_configs_key ON system_configs(key);
```

#### 6.2 用户配置表 (user_configs)

```sql
CREATE TABLE user_configs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category VARCHAR(50) NOT NULL,
    key VARCHAR(100) NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, category, key)
);

-- 索引
CREATE INDEX idx_user_configs_user_id ON user_configs(user_id);
CREATE INDEX idx_user_configs_category ON user_configs(category);
```

### 7. 日志和审计模块

#### 7.1 操作日志表 (audit_logs)

```sql
CREATE TABLE audit_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    request_id VARCHAR(100),
    session_id UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_resource_type ON audit_logs(resource_type);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);
```

#### 7.2 系统日志表 (system_logs)

```sql
CREATE TABLE system_logs (
    id BIGSERIAL PRIMARY KEY,
    level VARCHAR(10) NOT NULL CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    service VARCHAR(50) NOT NULL,
    module VARCHAR(50),
    message TEXT NOT NULL,
    details JSONB,
    trace_id VARCHAR(100),
    request_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_system_logs_level ON system_logs(level);
CREATE INDEX idx_system_logs_service ON system_logs(service);
CREATE INDEX idx_system_logs_created_at ON system_logs(created_at);
```

---

## InfluxDB 时序数据库设计

### 1. 行情数据 (Measurements)

#### 1.1 实时行情 (tickers)

```sql
-- Measurement: tickers
-- Tags: symbol, exchange
-- Fields: price, volume, bid, ask, high, low, change
-- Time: timestamp

CREATE MEASUREMENT tickers (
    time TIMESTAMP,
    symbol STRING,
    exchange STRING,
    price FLOAT,
    volume FLOAT,
    volume_24h FLOAT,
    bid FLOAT,
    ask FLOAT,
    bid_volume FLOAT,
    ask_volume FLOAT,
    high_24h FLOAT,
    low_24h FLOAT,
    change_24h FLOAT,
    change_24h_percent FLOAT,
    trades_count INTEGER
)
```

#### 1.2 K线数据 (klines)

```sql
-- Measurement: klines
-- Tags: symbol, exchange, timeframe
-- Fields: open, high, low, close, volume, trades
-- Time: timestamp

CREATE MEASUREMENT klines (
    time TIMESTAMP,
    symbol STRING,
    exchange STRING,
    timeframe STRING,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume FLOAT,
    quote_volume FLOAT,
    trades INTEGER,
    taker_buy_volume FLOAT,
    taker_buy_quote_volume FLOAT
)
```

#### 1.3 订单簿数据 (orderbook)

```sql
-- Measurement: orderbook
-- Tags: symbol, exchange, side
-- Fields: price, quantity, level
-- Time: timestamp

CREATE MEASUREMENT orderbook (
    time TIMESTAMP,
    symbol STRING,
    exchange STRING,
    side STRING,
    level INTEGER,
    price FLOAT,
    quantity FLOAT
)
```

### 2. 技术指标数据

#### 2.1 技术指标 (indicators)

```sql
-- Measurement: indicators
-- Tags: symbol, exchange, indicator_name, timeframe
-- Fields: value, signal, additional_values
-- Time: timestamp

CREATE MEASUREMENT indicators (
    time TIMESTAMP,
    symbol STRING,
    exchange STRING,
    indicator_name STRING,
    timeframe STRING,
    value FLOAT,
    signal STRING,
    additional_values JSONB
)
```

### 3. 选币评分数据

#### 3.1 策略评分 (strategy_scores)

```sql
-- Measurement: strategy_scores
-- Tags: symbol, exchange, strategy_name, user_id
-- Fields: score, rank, details
-- Time: timestamp

CREATE MEASUREMENT strategy_scores (
    time TIMESTAMP,
    symbol STRING,
    exchange STRING,
    strategy_name STRING,
    user_id STRING,
    score FLOAT,
    rank INTEGER,
    details JSONB
)
```

#### 3.2 综合评分 (coin_scores)

```sql
-- Measurement: coin_scores
-- Tags: symbol, exchange, user_id
-- Fields: total_score, rank, strategy_count
-- Time: timestamp

CREATE MEASUREMENT coin_scores (
    time TIMESTAMP,
    symbol STRING,
    exchange STRING,
    user_id STRING,
    total_score FLOAT,
    rank INTEGER,
    strategy_count INTEGER,
    market_cap FLOAT,
    volume_24h FLOAT
)
```

### 4. 系统监控数据

#### 4.1 服务性能 (service_metrics)

```sql
-- Measurement: service_metrics
-- Tags: service_name, instance, metric_type
-- Fields: value, unit
-- Time: timestamp

CREATE MEASUREMENT service_metrics (
    time TIMESTAMP,
    service_name STRING,
    instance STRING,
    metric_type STRING,
    value FLOAT,
    unit STRING
)
```

#### 4.2 API调用统计 (api_calls)

```sql
-- Measurement: api_calls
-- Tags: endpoint, method, status_code, user_id
-- Fields: response_time, request_size, response_size
-- Time: timestamp

CREATE MEASUREMENT api_calls (
    time TIMESTAMP,
    endpoint STRING,
    method STRING,
    status_code STRING,
    user_id STRING,
    response_time FLOAT,
    request_size INTEGER,
    response_size INTEGER
)
```

### 5. 数据保留策略

```sql
-- 创建数据保留策略
CREATE RETENTION POLICY "1_day" ON "market_data" DURATION 1d REPLICATION 1;
CREATE RETENTION POLICY "7_days" ON "market_data" DURATION 7d REPLICATION 1;
CREATE RETENTION POLICY "30_days" ON "market_data" DURATION 30d REPLICATION 1;
CREATE RETENTION POLICY "1_year" ON "market_data" DURATION 365d REPLICATION 1;
CREATE RETENTION POLICY "infinite" ON "market_data" DURATION INF REPLICATION 1 DEFAULT;

-- 连续查询用于数据聚合
CREATE CONTINUOUS QUERY "cq_klines_1h" ON "market_data"
BEGIN
  SELECT 
    first(open) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close) AS close,
    sum(volume) AS volume
  INTO "market_data"."1_year"."klines_1h"
  FROM "market_data"."1_day"."klines_1m"
  GROUP BY time(1h), symbol, exchange
END
```

---

## Redis 缓存数据库设计

### 1. 数据结构设计

#### 1.1 实时行情缓存

```redis
# 键名格式: ticker:{exchange}:{symbol}
# 数据类型: Hash
# TTL: 60秒

HMSET ticker:binance:BTC/USDT 
  price 43250.50
  volume 1234567.89
  change_24h 2.35
  change_24h_percent 5.76
  high_24h 43500.00
  low_24h 42000.00
  bid 43249.50
  ask 43251.00
  timestamp 1703123456789

EXPIRE ticker:binance:BTC/USDT 60
```

#### 1.2 K线数据缓存

```redis
# 键名格式: klines:{exchange}:{symbol}:{timeframe}
# 数据类型: List (最新100条)
# TTL: 300秒

LPUSH klines:binance:BTC/USDT:1h 
  '{"timestamp":1703123456789,"open":43000,"high":43300,"low":42900,"close":43250.50,"volume":123.45}'

LTRIM klines:binance:BTC/USDT:1h 0 99
EXPIRE klines:binance:BTC/USDT:1h 300
```

#### 1.3 选币结果缓存

```redis
# 键名格式: selections:{user_id}:latest
# 数据类型: Sorted Set (按评分排序)
# TTL: 1800秒

ZADD selections:123:latest 
  85.6 "BTC/USDT:binance"
  82.3 "ETH/USDT:binance"
  79.8 "ADA/USDT:binance"

EXPIRE selections:123:latest 1800
```

#### 1.4 用户会话缓存

```redis
# 键名格式: session:{session_id}
# 数据类型: Hash
# TTL: 3600秒

HMSET session:sess_123456789
  user_id 123
  username "john_doe"
  role "user"
  last_activity 1703123456789
  ip_address "192.168.1.100"

EXPIRE session:sess_123456789 3600
```

#### 1.5 API限流缓存

```redis
# 键名格式: rate_limit:{user_id}:{endpoint}
# 数据类型: String (计数器)
# TTL: 60秒

INCR rate_limit:123:/api/v1/ticker
EXPIRE rate_limit:123:/api/v1/ticker 60
```

### 2. 消息队列设计

#### 2.1 实时数据推送队列

```redis
# 使用 Pub/Sub 模式
# 频道格式: market_data:{type}:{symbol}

PUBLISH market_data:ticker:BTC/USDT 
  '{"price":43250.50,"volume":1234567.89,"timestamp":1703123456789}'

PUBLISH market_data:kline:BTC/USDT:1h
  '{"open":43000,"high":43300,"low":42900,"close":43250.50,"volume":123.45}'
```

#### 2.2 任务队列

```redis
# 使用 List 作为任务队列
# 队列名称: tasks:{service_name}

LPUSH tasks:coin_selection
  '{"task_id":"task_123","user_id":123,"strategy_ids":[1,2,3],"priority":"high"}'

LPUSH tasks:backtest
  '{"task_id":"bt_456","user_id":123,"start_date":"2023-01-01","end_date":"2023-12-01"}'
```

### 3. 缓存策略

#### 3.1 缓存层级

```
L1 Cache (应用内存) → L2 Cache (Redis) → L3 Cache (数据库)
```

#### 3.2 缓存更新策略

| 数据类型 | 更新策略 | TTL | 说明 |
|----------|----------|-----|------|
| 实时行情 | Write-Through | 60s | 实时更新，短TTL |
| K线数据 | Lazy Loading | 300s | 按需加载，中等TTL |
| 选币结果 | Write-Behind | 1800s | 异步更新，长TTL |
| 用户会话 | Write-Through | 3600s | 实时更新，会话TTL |
| 系统配置 | Cache-Aside | 86400s | 手动更新，长TTL |

---

## 数据库优化策略

### 1. PostgreSQL 优化

#### 1.1 索引优化

```sql
-- 复合索引
CREATE INDEX idx_coin_selections_user_score ON coin_selections(user_id, total_score DESC);
CREATE INDEX idx_backtest_trades_task_timestamp ON backtest_trades(task_id, timestamp);

-- 部分索引
CREATE INDEX idx_users_active ON users(id) WHERE status = 'active';
CREATE INDEX idx_strategies_enabled ON user_strategies(user_id) WHERE is_enabled = true;

-- 表达式索引
CREATE INDEX idx_users_email_lower ON users(lower(email));
CREATE INDEX idx_audit_logs_date ON audit_logs(date(created_at));
```

#### 1.2 分区表

```sql
-- 按时间分区的审计日志表
CREATE TABLE audit_logs_partitioned (
    LIKE audit_logs INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- 创建分区
CREATE TABLE audit_logs_2023_q1 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2023-01-01') TO ('2023-04-01');

CREATE TABLE audit_logs_2023_q2 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2023-04-01') TO ('2023-07-01');
```

#### 1.3 查询优化

```sql
-- 使用 EXPLAIN ANALYZE 分析查询
EXPLAIN ANALYZE 
SELECT cs.symbol, cs.total_score, cs.rank
FROM coin_selections cs
JOIN selection_tasks st ON cs.task_id = st.id
WHERE st.user_id = 123 
  AND cs.total_score > 80
ORDER BY cs.total_score DESC
LIMIT 20;

-- 使用窗口函数优化排名查询
SELECT 
    symbol,
    total_score,
    ROW_NUMBER() OVER (ORDER BY total_score DESC) as rank
FROM coin_selections
WHERE task_id = (SELECT id FROM selection_tasks WHERE user_id = 123 ORDER BY created_at DESC LIMIT 1);
```

### 2. InfluxDB 优化

#### 2.1 标签设计优化

```sql
-- 好的标签设计（基数较低）
symbol, exchange, timeframe

-- 避免高基数标签
-- 不要使用: timestamp, price, volume 作为标签
```

#### 2.2 查询优化

```sql
-- 使用时间范围过滤
SELECT mean(close) 
FROM klines 
WHERE time >= now() - 24h 
  AND symbol = 'BTC/USDT' 
  AND exchange = 'binance'
GROUP BY time(1h)

-- 使用聚合函数减少数据传输
SELECT 
    first(open) as open,
    max(high) as high,
    min(low) as low,
    last(close) as close,
    sum(volume) as volume
FROM klines
WHERE time >= now() - 7d
GROUP BY time(1h), symbol
```

### 3. Redis 优化

#### 3.1 内存优化

```redis
# 使用压缩数据结构
CONFIG SET hash-max-ziplist-entries 512
CONFIG SET hash-max-ziplist-value 64
CONFIG SET list-max-ziplist-size -2
CONFIG SET set-max-intset-entries 512

# 启用内存压缩
CONFIG SET rdbcompression yes
```

#### 3.2 持久化配置

```redis
# RDB 配置
save 900 1
save 300 10
save 60 10000

# AOF 配置
appendonly yes
appendfsync everysec
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

---

## 数据备份和恢复

### 1. PostgreSQL 备份策略

```bash
#!/bin/bash
# 每日全量备份
pg_dump -h localhost -U quant_user -d quant_system | gzip > backup_$(date +%Y%m%d).sql.gz

# 每小时增量备份（WAL归档）
archive_command = 'cp %p /backup/wal_archive/%f'
```

### 2. InfluxDB 备份策略

```bash
#!/bin/bash
# 备份数据库
influxd backup -portable -database market_data /backup/influxdb/$(date +%Y%m%d)

# 恢复数据库
influxd restore -portable -database market_data /backup/influxdb/20231201
```

### 3. Redis 备份策略

```bash
#!/bin/bash
# 创建 RDB 快照
redis-cli BGSAVE

# 复制 RDB 文件
cp /var/lib/redis/dump.rdb /backup/redis/dump_$(date +%Y%m%d_%H%M%S).rdb
```

---

*本数据库设计文档将随着系统开发进度持续更新和完善。*