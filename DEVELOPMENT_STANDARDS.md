# 量化交易系统开发规范

## 目录

1. [代码规范](#代码规范)
2. [架构设计原则](#架构设计原则)
3. [模块解耦规范](#模块解耦规范)
4. [命名规范](#命名规范)
5. [函数设计规范](#函数设计规范)
6. [类设计规范](#类设计规范)
7. [API设计规范](#api设计规范)
8. [数据库设计规范](#数据库设计规范)
9. [错误处理规范](#错误处理规范)
10. [测试规范](#测试规范)
11. [文档规范](#文档规范)
12. [版本控制规范](#版本控制规范)

---

## 代码规范

### Python 代码规范

#### 基础规范
- 遵循 **PEP 8** 标准
- 使用 **Black** 进行代码格式化
- 使用 **isort** 进行导入排序
- 使用 **flake8** 进行代码检查
- 使用 **mypy** 进行类型检查

#### 导入规范
```python
# 标准库导入
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

# 第三方库导入
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from sqlalchemy import Column, Integer, String

# 本地导入
from app.core.config import settings
from app.models.market_data import TickerData
from app.services.exchange_service import ExchangeService
```

#### 类型注解规范
```python
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal

# 函数类型注解
def calculate_rsi(
    prices: List[Decimal], 
    period: int = 14
) -> Optional[Decimal]:
    """计算RSI指标"""
    pass

# 类属性类型注解
class TickerData:
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: datetime
    
    def __init__(
        self, 
        symbol: str, 
        price: Decimal, 
        volume: Decimal
    ) -> None:
        self.symbol = symbol
        self.price = price
        self.volume = volume
        self.timestamp = datetime.utcnow()
```

### TypeScript 代码规范

#### 基础规范
- 使用 **ESLint** + **Prettier** 进行代码检查和格式化
- 严格模式 TypeScript 配置
- 使用 **Airbnb** 风格指南

#### 接口定义规范
```typescript
// 基础数据接口
interface TickerData {
  symbol: string;
  price: number;
  volume: number;
  change24h: number;
  timestamp: string;
}

// API 响应接口
interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  timestamp: string;
}

// 组件 Props 接口
interface CoinListProps {
  data: TickerData[];
  loading: boolean;
  onSelect: (symbol: string) => void;
  className?: string;
}
```

---

## 架构设计原则

### 1. SOLID 原则

#### 单一职责原则 (SRP)
```python
# ❌ 错误示例：一个类承担多个职责
class MarketDataProcessor:
    def fetch_data(self): pass
    def process_data(self): pass
    def save_to_database(self): pass
    def send_notification(self): pass

# ✅ 正确示例：职责分离
class DataFetcher:
    def fetch_data(self): pass

class DataProcessor:
    def process_data(self): pass

class DataStorage:
    def save_to_database(self): pass

class NotificationService:
    def send_notification(self): pass
```

#### 开闭原则 (OCP)
```python
# 策略模式实现开闭原则
from abc import ABC, abstractmethod

class SelectionStrategy(ABC):
    @abstractmethod
    def select_coins(self, market_data: List[TickerData]) -> List[str]:
        pass

class RSIStrategy(SelectionStrategy):
    def select_coins(self, market_data: List[TickerData]) -> List[str]:
        # RSI 策略实现
        pass

class VolumeStrategy(SelectionStrategy):
    def select_coins(self, market_data: List[TickerData]) -> List[str]:
        # 成交量策略实现
        pass

class CoinSelector:
    def __init__(self, strategy: SelectionStrategy):
        self.strategy = strategy
    
    def select(self, market_data: List[TickerData]) -> List[str]:
        return self.strategy.select_coins(market_data)
```

#### 依赖倒置原则 (DIP)
```python
# 使用依赖注入
from abc import ABC, abstractmethod

class DataRepository(ABC):
    @abstractmethod
    async def save_ticker_data(self, data: TickerData) -> bool:
        pass

class InfluxDBRepository(DataRepository):
    async def save_ticker_data(self, data: TickerData) -> bool:
        # InfluxDB 实现
        pass

class PostgreSQLRepository(DataRepository):
    async def save_ticker_data(self, data: TickerData) -> bool:
        # PostgreSQL 实现
        pass

class MarketDataService:
    def __init__(self, repository: DataRepository):
        self.repository = repository
    
    async def process_ticker(self, data: TickerData) -> bool:
        # 业务逻辑处理
        return await self.repository.save_ticker_data(data)
```

### 2. 分层架构

```
┌─────────────────────────────────────┐
│           Presentation Layer        │  # API 控制器、WebSocket 处理器
├─────────────────────────────────────┤
│           Application Layer         │  # 业务逻辑、用例实现
├─────────────────────────────────────┤
│             Domain Layer            │  # 领域模型、业务规则
├─────────────────────────────────────┤
│         Infrastructure Layer        │  # 数据访问、外部服务
└─────────────────────────────────────┘
```

---

## 模块解耦规范

### 1. 事件驱动架构

```python
# 事件定义
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

@dataclass
class DomainEvent:
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str

@dataclass
class TickerUpdatedEvent(DomainEvent):
    symbol: str
    price: Decimal
    volume: Decimal

# 事件发布器
class EventPublisher:
    def __init__(self):
        self._handlers: Dict[str, List[callable]] = {}
    
    def subscribe(self, event_type: str, handler: callable):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    async def publish(self, event: DomainEvent):
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            await handler(event)

# 事件处理器
class CoinSelectionEventHandler:
    async def handle_ticker_updated(self, event: TickerUpdatedEvent):
        # 处理行情更新事件
        pass

class NotificationEventHandler:
    async def handle_ticker_updated(self, event: TickerUpdatedEvent):
        # 发送通知
        pass
```

### 2. 消息队列解耦

```python
# 消息生产者
class MessageProducer:
    def __init__(self, kafka_client):
        self.kafka_client = kafka_client
    
    async def send_ticker_update(self, ticker_data: TickerData):
        message = {
            "type": "ticker_update",
            "data": ticker_data.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.kafka_client.send("market_data", message)

# 消息消费者
class MessageConsumer:
    def __init__(self, kafka_client):
        self.kafka_client = kafka_client
    
    async def consume_ticker_updates(self):
        async for message in self.kafka_client.consume("market_data"):
            await self.process_ticker_update(message)
    
    async def process_ticker_update(self, message: dict):
        # 处理行情更新消息
        pass
```

### 3. 接口隔离

```python
# 细粒度接口设计
class TickerDataReader(ABC):
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Optional[TickerData]:
        pass

class TickerDataWriter(ABC):
    @abstractmethod
    async def save_ticker(self, data: TickerData) -> bool:
        pass

class HistoricalDataReader(ABC):
    @abstractmethod
    async def get_klines(
        self, 
        symbol: str, 
        interval: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[KlineData]:
        pass

# 服务只依赖需要的接口
class CoinSelectionService:
    def __init__(self, ticker_reader: TickerDataReader):
        self.ticker_reader = ticker_reader
    
    async def analyze_coin(self, symbol: str) -> CoinScore:
        ticker = await self.ticker_reader.get_ticker(symbol)
        # 分析逻辑
        pass
```

---

## 命名规范

### 1. Python 命名规范

```python
# 常量：全大写，下划线分隔
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.binance.com"

# 变量：小写，下划线分隔
ticker_data = get_ticker_data()
market_cap_threshold = 1000000
current_timestamp = datetime.utcnow()

# 函数：小写，下划线分隔，动词开头
def calculate_moving_average(prices: List[Decimal], period: int) -> Decimal:
    pass

def fetch_market_data(exchange: str, symbol: str) -> TickerData:
    pass

def validate_strategy_config(config: dict) -> bool:
    pass

# 类：帕斯卡命名法，名词
class MarketDataCollector:
    pass

class CoinSelectionStrategy:
    pass

class DatabaseConnection:
    pass

# 方法：小写，下划线分隔
class ExchangeManager:
    def connect_to_exchange(self) -> bool:
        pass
    
    def get_account_balance(self) -> Dict[str, Decimal]:
        pass
    
    def place_order(self, order: OrderRequest) -> OrderResponse:
        pass

# 私有方法：下划线开头
class DataProcessor:
    def process_data(self, data: List[dict]) -> List[TickerData]:
        cleaned_data = self._clean_data(data)
        return self._transform_data(cleaned_data)
    
    def _clean_data(self, data: List[dict]) -> List[dict]:
        pass
    
    def _transform_data(self, data: List[dict]) -> List[TickerData]:
        pass
```

### 2. TypeScript 命名规范

```typescript
// 常量：全大写，下划线分隔
const MAX_RETRY_COUNT = 3;
const API_ENDPOINTS = {
  MARKET_DATA: '/api/v1/market-data',
  COIN_SELECTION: '/api/v1/coin-selection'
};

// 变量：驼峰命名法
const tickerData = await fetchTickerData();
const marketCapThreshold = 1000000;
const currentTimestamp = new Date();

// 函数：驼峰命名法，动词开头
function calculateMovingAverage(prices: number[], period: number): number {
  // 实现
}

function fetchMarketData(exchange: string, symbol: string): Promise<TickerData> {
  // 实现
}

function validateStrategyConfig(config: StrategyConfig): boolean {
  // 实现
}

// 接口：帕斯卡命名法，名词
interface TickerData {
  symbol: string;
  price: number;
  volume: number;
}

interface CoinSelectionConfig {
  strategy: string;
  parameters: Record<string, any>;
}

// 类：帕斯卡命名法，名词
class MarketDataService {
  private apiClient: ApiClient;
  
  constructor(apiClient: ApiClient) {
    this.apiClient = apiClient;
  }
  
  public async getTickerData(symbol: string): Promise<TickerData> {
    // 实现
  }
  
  private async validateSymbol(symbol: string): Promise<boolean> {
    // 实现
  }
}

// 组件：帕斯卡命名法
const CoinListTable: React.FC<CoinListProps> = ({ data, loading }) => {
  return (
    // JSX
  );
};

// Hook：use 开头，驼峰命名法
function useMarketData(symbol: string) {
  const [data, setData] = useState<TickerData | null>(null);
  const [loading, setLoading] = useState(false);
  
  // 实现
  
  return { data, loading };
}
```

### 3. 数据库命名规范

```sql
-- 表名：小写，下划线分隔，复数形式
CREATE TABLE ticker_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE coin_selection_results (
    id BIGSERIAL PRIMARY KEY,
    strategy_id INTEGER NOT NULL,
    selected_coins JSONB NOT NULL,
    score DECIMAL(10, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 索引名：表名_字段名_idx
CREATE INDEX ticker_data_symbol_idx ON ticker_data(symbol);
CREATE INDEX ticker_data_created_at_idx ON ticker_data(created_at);

-- 外键约束：表名_字段名_fkey
ALTER TABLE coin_selection_results 
ADD CONSTRAINT coin_selection_results_strategy_id_fkey 
FOREIGN KEY (strategy_id) REFERENCES strategies(id);
```

---

## 函数设计规范

### 1. 函数职责单一

```python
# ❌ 错误示例：函数职责过多
def process_market_data_and_select_coins(exchange: str) -> List[str]:
    # 获取数据
    data = fetch_data_from_exchange(exchange)
    # 清洗数据
    cleaned_data = clean_data(data)
    # 计算指标
    indicators = calculate_indicators(cleaned_data)
    # 选币
    selected_coins = select_coins_by_rsi(indicators)
    # 发送通知
    send_notification(selected_coins)
    return selected_coins

# ✅ 正确示例：职责分离
def fetch_and_clean_market_data(exchange: str) -> List[TickerData]:
    """获取并清洗市场数据"""
    raw_data = fetch_data_from_exchange(exchange)
    return clean_ticker_data(raw_data)

def calculate_technical_indicators(data: List[TickerData]) -> Dict[str, Any]:
    """计算技术指标"""
    return {
        'rsi': calculate_rsi(data),
        'macd': calculate_macd(data),
        'volume_profile': calculate_volume_profile(data)
    }

def select_coins_by_strategy(
    data: List[TickerData], 
    strategy: SelectionStrategy
) -> List[str]:
    """根据策略选择币种"""
    return strategy.select_coins(data)
```

### 2. 参数设计规范

```python
# 使用类型注解和默认值
def calculate_moving_average(
    prices: List[Decimal],
    period: int = 20,
    ma_type: str = "simple"
) -> Optional[Decimal]:
    """计算移动平均线
    
    Args:
        prices: 价格列表
        period: 计算周期，默认20
        ma_type: 移动平均类型，支持 'simple', 'exponential'
    
    Returns:
        移动平均值，如果数据不足返回 None
    
    Raises:
        ValueError: 当 period 小于等于 0 时
    """
    if period <= 0:
        raise ValueError("Period must be greater than 0")
    
    if len(prices) < period:
        return None
    
    if ma_type == "simple":
        return sum(prices[-period:]) / period
    elif ma_type == "exponential":
        return calculate_ema(prices, period)
    else:
        raise ValueError(f"Unsupported MA type: {ma_type}")

# 使用数据类传递复杂参数
@dataclass
class StrategyConfig:
    name: str
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    volume_threshold: Decimal = Decimal('1000000')
    max_positions: int = 10

def execute_strategy(
    market_data: List[TickerData],
    config: StrategyConfig
) -> SelectionResult:
    """执行选币策略"""
    pass
```

### 3. 返回值规范

```python
# 使用明确的返回类型
from typing import Tuple, Optional, Union

# 成功/失败结果
class Result:
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error
    
    @classmethod
    def success(cls, data: Any) -> 'Result':
        return cls(True, data=data)
    
    @classmethod
    def failure(cls, error: str) -> 'Result':
        return cls(False, error=error)

def fetch_ticker_data(symbol: str) -> Result[TickerData]:
    """获取行情数据"""
    try:
        data = api_client.get_ticker(symbol)
        return Result.success(data)
    except Exception as e:
        return Result.failure(str(e))

# 使用 Optional 处理可能为空的返回值
def find_coin_by_symbol(symbol: str) -> Optional[CoinInfo]:
    """根据符号查找币种信息"""
    pass

# 使用 Tuple 返回多个值
def calculate_support_resistance(
    prices: List[Decimal]
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """计算支撑位和阻力位
    
    Returns:
        (支撑位, 阻力位) 的元组
    """
    pass
```

---

## 类设计规范

### 1. 类的职责设计

```python
# 数据类：只包含数据和简单的数据操作
@dataclass
class TickerData:
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'price': float(self.price),
            'volume': float(self.volume),
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TickerData':
        return cls(
            symbol=data['symbol'],
            price=Decimal(str(data['price'])),
            volume=Decimal(str(data['volume'])),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

# 服务类：包含业务逻辑
class MarketDataService:
    def __init__(
        self, 
        repository: DataRepository,
        event_publisher: EventPublisher
    ):
        self.repository = repository
        self.event_publisher = event_publisher
    
    async def update_ticker_data(self, data: TickerData) -> bool:
        """更新行情数据"""
        try:
            # 保存数据
            success = await self.repository.save_ticker_data(data)
            
            if success:
                # 发布事件
                event = TickerUpdatedEvent(
                    event_type="ticker_updated",
                    data=data.to_dict(),
                    timestamp=datetime.utcnow(),
                    source="market_data_service",
                    symbol=data.symbol,
                    price=data.price,
                    volume=data.volume
                )
                await self.event_publisher.publish(event)
            
            return success
        except Exception as e:
            logger.error(f"Failed to update ticker data: {e}")
            return False

# 工厂类：创建对象
class ExchangeFactory:
    _exchanges = {
        'binance': BinanceExchange,
        'okx': OKXExchange,
        'huobi': HuobiExchange
    }
    
    @classmethod
    def create_exchange(cls, exchange_name: str, config: dict) -> Exchange:
        if exchange_name not in cls._exchanges:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        
        exchange_class = cls._exchanges[exchange_name]
        return exchange_class(config)
    
    @classmethod
    def get_supported_exchanges(cls) -> List[str]:
        return list(cls._exchanges.keys())
```

### 2. 继承和组合

```python
# 优先使用组合而不是继承
class CoinSelector:
    def __init__(
        self,
        data_source: MarketDataSource,
        strategy: SelectionStrategy,
        risk_manager: RiskManager
    ):
        self.data_source = data_source
        self.strategy = strategy
        self.risk_manager = risk_manager
    
    async def select_coins(self) -> SelectionResult:
        # 获取数据
        market_data = await self.data_source.get_market_data()
        
        # 执行策略
        candidates = self.strategy.select_coins(market_data)
        
        # 风险控制
        filtered_coins = self.risk_manager.filter_by_risk(candidates)
        
        return SelectionResult(
            coins=filtered_coins,
            timestamp=datetime.utcnow(),
            strategy=self.strategy.name
        )

# 使用抽象基类定义接口
from abc import ABC, abstractmethod

class SelectionStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def select_coins(self, market_data: List[TickerData]) -> List[str]:
        pass
    
    @abstractmethod
    def get_parameters(self) -> dict:
        pass

class RSIStrategy(SelectionStrategy):
    def __init__(self, period: int = 14, oversold: int = 30):
        self.period = period
        self.oversold = oversold
    
    @property
    def name(self) -> str:
        return "RSI Strategy"
    
    def select_coins(self, market_data: List[TickerData]) -> List[str]:
        # RSI 策略实现
        pass
    
    def get_parameters(self) -> dict:
        return {
            'period': self.period,
            'oversold': self.oversold
        }
```

---

## API设计规范

### 1. RESTful API 设计

```python
# 资源命名：使用复数名词
# GET /api/v1/tickers - 获取所有行情数据
# GET /api/v1/tickers/{symbol} - 获取特定币种行情
# POST /api/v1/strategies - 创建策略
# PUT /api/v1/strategies/{id} - 更新策略
# DELETE /api/v1/strategies/{id} - 删除策略

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional

router = APIRouter(prefix="/api/v1", tags=["market-data"])

@router.get("/tickers", response_model=List[TickerResponse])
async def get_tickers(
    exchange: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    market_data_service: MarketDataService = Depends(get_market_data_service)
) -> List[TickerResponse]:
    """获取行情数据列表
    
    Args:
        exchange: 交易所名称，可选
        limit: 返回数量限制，默认100
        offset: 偏移量，默认0
    
    Returns:
        行情数据列表
    """
    try:
        tickers = await market_data_service.get_tickers(
            exchange=exchange,
            limit=limit,
            offset=offset
        )
        return [TickerResponse.from_domain(ticker) for ticker in tickers]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tickers/{symbol}", response_model=TickerResponse)
async def get_ticker(
    symbol: str,
    market_data_service: MarketDataService = Depends(get_market_data_service)
) -> TickerResponse:
    """获取特定币种行情数据"""
    ticker = await market_data_service.get_ticker(symbol)
    if not ticker:
        raise HTTPException(status_code=404, detail="Ticker not found")
    return TickerResponse.from_domain(ticker)

@router.post("/strategies", response_model=StrategyResponse)
async def create_strategy(
    request: CreateStrategyRequest,
    strategy_service: StrategyService = Depends(get_strategy_service)
) -> StrategyResponse:
    """创建选币策略"""
    try:
        strategy = await strategy_service.create_strategy(
            name=request.name,
            config=request.config,
            user_id=request.user_id
        )
        return StrategyResponse.from_domain(strategy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. 请求/响应模型

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from decimal import Decimal
from datetime import datetime

# 请求模型
class CreateStrategyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    config: Dict[str, Any] = Field(...)
    user_id: int = Field(..., gt=0)
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
    
    @validator('config')
    def validate_config(cls, v):
        required_fields = ['strategy_type', 'parameters']
        for field in required_fields:
            if field not in v:
                raise ValueError(f'Missing required field: {field}')
        return v

# 响应模型
class TickerResponse(BaseModel):
    symbol: str
    price: Decimal
    volume: Decimal
    change_24h: Decimal
    timestamp: datetime
    
    @classmethod
    def from_domain(cls, ticker: TickerData) -> 'TickerResponse':
        return cls(
            symbol=ticker.symbol,
            price=ticker.price,
            volume=ticker.volume,
            change_24h=ticker.change_24h,
            timestamp=ticker.timestamp
        )
    
    class Config:
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }

class ApiResponse(BaseModel):
    """统一API响应格式"""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def success_response(cls, data: Any = None, message: str = None):
        return cls(success=True, data=data, message=message)
    
    @classmethod
    def error_response(cls, message: str, data: Any = None):
        return cls(success=False, message=message, data=data)
```

---

## 错误处理规范

### 1. 异常层次设计

```python
# 基础异常类
class QuantTradingException(Exception):
    """量化交易系统基础异常"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

# 业务异常
class BusinessException(QuantTradingException):
    """业务逻辑异常"""
    pass

class ValidationException(BusinessException):
    """数据验证异常"""
    pass

class StrategyException(BusinessException):
    """策略执行异常"""
    pass

# 技术异常
class TechnicalException(QuantTradingException):
    """技术异常"""
    pass

class DatabaseException(TechnicalException):
    """数据库异常"""
    pass

class ExchangeAPIException(TechnicalException):
    """交易所API异常"""
    def __init__(self, message: str, exchange: str, status_code: int = None):
        self.exchange = exchange
        self.status_code = status_code
        super().__init__(message, f"EXCHANGE_API_ERROR_{status_code}")

# 具体异常
class InsufficientDataException(ValidationException):
    """数据不足异常"""
    def __init__(self, required_count: int, actual_count: int):
        message = f"Insufficient data: required {required_count}, got {actual_count}"
        super().__init__(message, "INSUFFICIENT_DATA")
        self.required_count = required_count
        self.actual_count = actual_count

class StrategyNotFound(StrategyException):
    """策略未找到异常"""
    def __init__(self, strategy_id: int):
        message = f"Strategy not found: {strategy_id}"
        super().__init__(message, "STRATEGY_NOT_FOUND")
        self.strategy_id = strategy_id
```

### 2. 错误处理模式

```python
# 使用装饰器处理异常
from functools import wraps
import logging

def handle_exceptions(logger: logging.Logger = None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ValidationException as e:
                if logger:
                    logger.warning(f"Validation error in {func.__name__}: {e}")
                raise
            except BusinessException as e:
                if logger:
                    logger.error(f"Business error in {func.__name__}: {e}")
                raise
            except Exception as e:
                if logger:
                    logger.exception(f"Unexpected error in {func.__name__}: {e}")
                raise TechnicalException(f"Internal error: {str(e)}")
        return wrapper
    return decorator

# 使用示例
class CoinSelectionService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @handle_exceptions()
    async def select_coins(self, strategy_id: int) -> List[str]:
        # 验证策略存在
        strategy = await self.get_strategy(strategy_id)
        if not strategy:
            raise StrategyNotFound(strategy_id)
        
        # 获取市场数据
        market_data = await self.get_market_data()
        if len(market_data) < 10:
            raise InsufficientDataException(10, len(market_data))
        
        # 执行选币逻辑
        return strategy.select_coins(market_data)
```

### 3. 全局异常处理

```python
# FastAPI 全局异常处理
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error_code": exc.error_code,
            "message": exc.message,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(BusinessException)
async def business_exception_handler(request: Request, exc: BusinessException):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error_code": exc.error_code,
            "message": exc.message,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(TechnicalException)
async def technical_exception_handler(request: Request, exc: TechnicalException):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": exc.error_code or "INTERNAL_ERROR",
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

---

## 测试规范

### 1. 单元测试

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime

class TestCoinSelectionService:
    @pytest.fixture
    def mock_repository(self):
        return Mock(spec=DataRepository)
    
    @pytest.fixture
    def mock_event_publisher(self):
        return Mock(spec=EventPublisher)
    
    @pytest.fixture
    def service(self, mock_repository, mock_event_publisher):
        return CoinSelectionService(
            repository=mock_repository,
            event_publisher=mock_event_publisher
        )
    
    @pytest.fixture
    def sample_ticker_data(self):
        return TickerData(
            symbol="BTCUSDT",
            price=Decimal("50000.00"),
            volume=Decimal("1000.00"),
            timestamp=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_select_coins_success(self, service, sample_ticker_data):
        # Arrange
        market_data = [sample_ticker_data]
        strategy = Mock(spec=SelectionStrategy)
        strategy.select_coins.return_value = ["BTCUSDT"]
        
        # Act
        result = await service.select_coins_by_strategy(market_data, strategy)
        
        # Assert
        assert result == ["BTCUSDT"]
        strategy.select_coins.assert_called_once_with(market_data)
    
    @pytest.mark.asyncio
    async def test_select_coins_insufficient_data(self, service):
        # Arrange
        market_data = []  # 空数据
        strategy = Mock(spec=SelectionStrategy)
        
        # Act & Assert
        with pytest.raises(InsufficientDataException) as exc_info:
            await service.select_coins_by_strategy(market_data, strategy)
        
        assert exc_info.value.required_count == 10
        assert exc_info.value.actual_count == 0
    
    @pytest.mark.asyncio
    async def test_update_ticker_data_success(self, service, sample_ticker_data, mock_repository, mock_event_publisher):
        # Arrange
        mock_repository.save_ticker_data.return_value = True
        mock_event_publisher.publish = AsyncMock()
        
        # Act
        result = await service.update_ticker_data(sample_ticker_data)
        
        # Assert
        assert result is True
        mock_repository.save_ticker_data.assert_called_once_with(sample_ticker_data)
        mock_event_publisher.publish.assert_called_once()
```

### 2. 集成测试

```python
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient

class TestMarketDataAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    def test_get_tickers_success(self, client):
        # Act
        response = client.get("/api/v1/tickers")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["success"] is True
        assert "data" in data
        assert isinstance(data["data"], list)
    
    def test_get_ticker_not_found(self, client):
        # Act
        response = client.get("/api/v1/tickers/INVALID")
        
        # Assert
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "Ticker not found" in data["message"]
    
    @pytest.mark.asyncio
    async def test_create_strategy_success(self, async_client):
        # Arrange
        strategy_data = {
            "name": "Test Strategy",
            "description": "Test description",
            "config": {
                "strategy_type": "rsi",
                "parameters": {
                    "period": 14,
                    "oversold": 30
                }
            },
            "user_id": 1
        }
        
        # Act
        response = await async_client.post("/api/v1/strategies", json=strategy_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == "Test Strategy"
```

### 3. 性能测试

```python
import pytest
import time
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    @pytest.mark.performance
    def test_ticker_data_processing_performance(self):
        # Arrange
        large_dataset = generate_large_ticker_dataset(10000)
        processor = DataProcessor()
        
        # Act
        start_time = time.time()
        result = processor.process_ticker_data(large_dataset)
        end_time = time.time()
        
        # Assert
        processing_time = end_time - start_time
        assert processing_time < 1.0  # 应该在1秒内完成
        assert len(result) == len(large_dataset)
    
    @pytest.mark.performance
    def test_concurrent_api_requests(self):
        # Arrange
        client = TestClient(app)
        num_requests = 100
        
        def make_request():
            return client.get("/api/v1/tickers")
        
        # Act
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            responses = [future.result() for future in futures]
        end_time = time.time()
        
        # Assert
        total_time = end_time - start_time
        assert total_time < 5.0  # 100个请求应该在5秒内完成
        assert all(response.status_code == 200 for response in responses)
```

---

## 文档规范

### 1. 代码文档

```python
def calculate_rsi(
    prices: List[Decimal], 
    period: int = 14
) -> Optional[Decimal]:
    """计算相对强弱指数(RSI)
    
    RSI是一个动量振荡器，用于衡量价格变动的速度和变化。
    RSI值在0到100之间波动，通常70以上被认为是超买，30以下被认为是超卖。
    
    Args:
        prices: 价格序列，按时间顺序排列（最新的在最后）
        period: 计算周期，默认14个周期
    
    Returns:
        RSI值，如果数据不足则返回None
        
    Raises:
        ValueError: 当period小于等于0时
        TypeError: 当prices不是List[Decimal]类型时
    
    Example:
        >>> prices = [Decimal('100'), Decimal('102'), Decimal('101'), Decimal('103')]
        >>> rsi = calculate_rsi(prices, period=3)
        >>> print(f"RSI: {rsi}")
        RSI: 66.67
    
    Note:
        - 需要至少period+1个价格数据点才能计算RSI
        - 使用Wilder's smoothing方法计算平均收益和损失
        - 第一个RSI值通常在第period+1个数据点计算
    
    References:
        - Wilder, J. Welles (1978). New Concepts in Technical Trading Systems
        - https://en.wikipedia.org/wiki/Relative_strength_index
    """
    if period <= 0:
        raise ValueError("Period must be greater than 0")
    
    if not isinstance(prices, list) or not all(isinstance(p, Decimal) for p in prices):
        raise TypeError("Prices must be a list of Decimal values")
    
    if len(prices) < period + 1:
        return None
    
    # 计算价格变化
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    # 分离收益和损失
    gains = [max(change, Decimal('0')) for change in changes]
    losses = [abs(min(change, Decimal('0'))) for change in changes]
    
    # 计算平均收益和损失
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return Decimal('100')
    
    rs = avg_gain / avg_loss
    rsi = Decimal('100') - (Decimal('100') / (Decimal('1') + rs))
    
    return rsi.quantize(Decimal('0.01'))
```

### 2. API文档

```python
from fastapi import APIRouter
from fastapi.openapi.docs import get_swagger_ui_html

router = APIRouter()

@router.get(
    "/tickers/{symbol}",
    response_model=TickerResponse,
    summary="获取币种行情数据",
    description="根据交易对符号获取实时行情数据，包括价格、成交量、涨跌幅等信息",
    response_description="返回指定币种的详细行情数据",
    tags=["市场数据"],
    responses={
        200: {
            "description": "成功获取行情数据",
            "content": {
                "application/json": {
                    "example": {
                        "symbol": "BTCUSDT",
                        "price": 50000.00,
                        "volume": 1000.50,
                        "change_24h": 2.5,
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                }
            }
        },
        404: {
            "description": "币种未找到",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "Ticker not found",
                        "error_code": "TICKER_NOT_FOUND"
                    }
                }
            }
        }
    }
)
async def get_ticker(
    symbol: str = Path(
        ..., 
        description="交易对符号，如BTCUSDT", 
        example="BTCUSDT",
        regex="^[A-Z0-9]+$"
    ),
    exchange: Optional[str] = Query(
        None, 
        description="交易所名称，不指定则返回默认交易所数据",
        example="binance"
    )
) -> TickerResponse:
    """获取指定币种的实时行情数据
    
    此接口返回指定交易对的最新行情信息，包括：
    - 当前价格
    - 24小时成交量
    - 24小时价格变化百分比
    - 数据更新时间戳
    
    支持的交易所：
    - binance: 币安
    - okx: OKX
    - huobi: 火币
    
    限制：
    - 每分钟最多100次请求
    - 数据延迟通常在1-3秒内
    """
    pass
```

---

## 版本控制规范

### 1. Git 提交规范

```bash
# 提交消息格式
<type>(<scope>): <subject>

<body>

<footer>

# 类型说明
feat: 新功能
fix: 修复bug
docs: 文档更新
style: 代码格式调整（不影响功能）
refactor: 代码重构
test: 测试相关
chore: 构建过程或辅助工具的变动
perf: 性能优化

# 示例
feat(market-data): add real-time ticker data collection

- Implement WebSocket connection to Binance API
- Add data validation and error handling
- Support multiple trading pairs subscription

Closes #123

fix(coin-selection): fix RSI calculation overflow issue

- Handle edge case when all price changes are positive
- Add proper decimal precision handling
- Update unit tests for edge cases

Fixes #456

docs(api): update API documentation for v1.2

- Add new endpoints documentation
- Update response examples
- Fix typos in parameter descriptions

refactor(database): optimize query performance

- Add database indexes for frequently queried columns
- Refactor complex queries to use joins instead of subqueries
- Remove unused database connections

Performance improvement: 40% faster query execution
```

### 2. 分支管理策略

```bash
# 分支命名规范
main                    # 主分支，生产环境代码
develop                 # 开发分支，集成最新功能
feature/feature-name    # 功能分支
fix/bug-description     # 修复分支
hotfix/critical-fix     # 热修复分支
release/v1.2.0         # 发布分支

# 功能开发流程
git checkout develop
git pull origin develop
git checkout -b feature/coin-selection-rsi-strategy

# 开发完成后
git add .
git commit -m "feat(coin-selection): implement RSI-based selection strategy"
git push origin feature/coin-selection-rsi-strategy

# 创建Pull Request到develop分支

# 发布流程
git checkout develop
git pull origin develop
git checkout -b release/v1.2.0

# 更新版本号，修复发现的问题
git commit -m "chore(release): bump version to 1.2.0"
git push origin release/v1.2.0

# 合并到main和develop
git checkout main
git merge release/v1.2.0
git tag v1.2.0
git push origin main --tags

git checkout develop
git merge release/v1.2.0
git push origin develop
```

### 3. 代码审查规范

```markdown
# Pull Request 模板

## 变更描述
简要描述本次变更的内容和目的

## 变更类型
- [ ] 新功能
- [ ] Bug修复
- [ ] 文档更新
- [ ] 代码重构
- [ ] 性能优化
- [ ] 测试改进

## 测试
- [ ] 单元测试已通过
- [ ] 集成测试已通过
- [ ] 手动测试已完成
- [ ] 性能测试已完成（如适用）

## 检查清单
- [ ] 代码遵循项目编码规范
- [ ] 已添加必要的测试用例
- [ ] 已更新相关文档
- [ ] 已考虑向后兼容性
- [ ] 已处理错误情况
- [ ] 已添加适当的日志记录

## 相关Issue
Closes #123
Related to #456

## 截图（如适用）

## 额外说明
```

---

## 总结

本开发规范文档涵盖了量化交易系统开发的各个方面：

1. **代码质量保证**：通过严格的命名规范、类型注解、代码格式化确保代码可读性和维护性
2. **架构设计原则**：遵循SOLID原则，采用分层架构和事件驱动模式实现模块解耦
3. **错误处理机制**：建立完整的异常层次和处理机制，确保系统稳定性
4. **测试策略**：包含单元测试、集成测试、性能测试的完整测试体系
5. **文档规范**：详细的代码文档和API文档，便于团队协作和维护
6. **版本控制**：规范的Git工作流程和代码审查机制

遵循这些规范将帮助团队：
- 提高代码质量和可维护性
- 减少重复开发和技术债务
- 提升团队协作效率
- 确保系统的稳定性和可扩展性
- 便于新成员快速上手

建议在项目开始前，团队成员共同学习和讨论这些规范，并在开发过程中严格执行。