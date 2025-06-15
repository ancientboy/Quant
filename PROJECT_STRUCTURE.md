# 项目目录结构

## 整体项目结构

```
Quant/
├── README.md                           # 项目说明文档
├── DEVELOPMENT_PLAN.md                 # 开发计划文档
├── TECHNICAL_ARCHITECTURE.md           # 技术架构文档
├── PROJECT_STRUCTURE.md               # 项目结构说明（本文档）
├── docker-compose.yml                 # Docker编排文件
├── docker-compose.dev.yml             # 开发环境Docker编排
├── .env.example                       # 环境变量示例
├── .gitignore                         # Git忽略文件
├── requirements.txt                   # Python依赖（已存在）
├── Makefile                           # 构建和部署脚本
│
├── docs/                              # 文档目录
│   ├── api/                          # API文档
│   │   ├── market-data-api.md
│   │   └── coin-selection-api.md
│   ├── deployment/                   # 部署文档
│   │   ├── docker-deployment.md
│   │   └── k8s-deployment.md
│   └── user-guide/                   # 用户指南
│       ├── web-interface.md
│       └── api-usage.md
│
├── scripts/                           # 脚本目录
│   ├── setup/                        # 安装脚本
│   │   ├── install-dependencies.sh
│   │   └── setup-database.sh
│   ├── deployment/                   # 部署脚本
│   │   ├── deploy-dev.sh
│   │   └── deploy-prod.sh
│   └── maintenance/                  # 维护脚本
│       ├── backup-database.sh
│       └── cleanup-logs.sh
│
├── config/                           # 配置文件目录
│   ├── development/                  # 开发环境配置
│   │   ├── database.yaml
│   │   ├── redis.yaml
│   │   └── strategies.yaml
│   ├── production/                   # 生产环境配置
│   │   ├── database.yaml
│   │   ├── redis.yaml
│   │   └── strategies.yaml
│   └── nginx/                        # Nginx配置
│       ├── nginx.conf
│       └── ssl/
│
├── monitoring/                       # 监控配置
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── alert_rules.yml
│   ├── grafana/
│   │   ├── dashboards/
│   │   └── provisioning/
│   └── logs/
│       └── logstash.conf
│
├── sql/                              # 数据库脚本
│   ├── migrations/                   # 数据库迁移
│   │   ├── 001_initial_schema.sql
│   │   ├── 002_add_strategies.sql
│   │   └── 003_add_indexes.sql
│   ├── seeds/                        # 初始数据
│   │   ├── exchanges.sql
│   │   └── default_strategies.sql
│   └── init.sql                      # 初始化脚本
│
├── tests/                            # 测试目录
│   ├── unit/                         # 单元测试
│   │   ├── market_data/
│   │   └── coin_selection/
│   ├── integration/                  # 集成测试
│   │   ├── test_api_integration.py
│   │   └── test_database_integration.py
│   ├── e2e/                          # 端到端测试
│   │   └── test_full_workflow.py
│   └── fixtures/                     # 测试数据
│       ├── sample_market_data.json
│       └── sample_strategies.yaml
│
├── market-data-server/               # 行情数据服务器
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                       # 应用入口
│   ├── config.py                     # 配置管理
│   │
│   ├── app/                          # 应用核心
│   │   ├── __init__.py
│   │   ├── api/                      # API路由
│   │   │   ├── __init__.py
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ticker.py
│   │   │   │   ├── klines.py
│   │   │   │   ├── orderbook.py
│   │   │   │   └── websocket.py
│   │   │   └── dependencies.py       # API依赖注入
│   │   │
│   │   ├── core/                     # 核心业务逻辑
│   │   │   ├── __init__.py
│   │   │   ├── exchange_manager.py   # 交易所管理
│   │   │   ├── data_collector.py     # 数据收集
│   │   │   ├── data_processor.py     # 数据处理
│   │   │   ├── data_storage.py       # 数据存储
│   │   │   └── websocket_manager.py  # WebSocket管理
│   │   │
│   │   ├── models/                   # 数据模型
│   │   │   ├── __init__.py
│   │   │   ├── ticker.py
│   │   │   ├── kline.py
│   │   │   ├── orderbook.py
│   │   │   └── exchange.py
│   │   │
│   │   ├── services/                 # 服务层
│   │   │   ├── __init__.py
│   │   │   ├── market_data_service.py
│   │   │   ├── exchange_service.py
│   │   │   └── notification_service.py
│   │   │
│   │   └── utils/                    # 工具函数
│   │       ├── __init__.py
│   │       ├── logger.py
│   │       ├── validators.py
│   │       ├── formatters.py
│   │       └── exceptions.py
│   │
│   └── tests/                        # 服务专用测试
│       ├── test_exchange_manager.py
│       ├── test_data_collector.py
│       └── test_api_endpoints.py
│
├── coin-selection-system/            # 选币系统
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                       # 应用入口
│   ├── config.py                     # 配置管理
│   │
│   ├── app/                          # 应用核心
│   │   ├── __init__.py
│   │   ├── api/                      # API路由
│   │   │   ├── __init__.py
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── strategies.py
│   │   │   │   ├── selections.py
│   │   │   │   ├── backtest.py
│   │   │   │   └── analysis.py
│   │   │   └── dependencies.py
│   │   │
│   │   ├── core/                     # 核心业务逻辑
│   │   │   ├── __init__.py
│   │   │   ├── strategy_engine.py    # 策略引擎
│   │   │   ├── technical_analyzer.py # 技术分析
│   │   │   ├── fundamental_analyzer.py # 基本面分析
│   │   │   ├── backtest_engine.py    # 回测引擎
│   │   │   └── risk_manager.py       # 风险管理
│   │   │
│   │   ├── strategies/               # 策略实现
│   │   │   ├── __init__.py
│   │   │   ├── base_strategy.py      # 策略基类
│   │   │   ├── technical/            # 技术分析策略
│   │   │   │   ├── __init__.py
│   │   │   │   ├── rsi_strategy.py
│   │   │   │   ├── macd_strategy.py
│   │   │   │   ├── volume_strategy.py
│   │   │   │   └── momentum_strategy.py
│   │   │   ├── fundamental/          # 基本面策略
│   │   │   │   ├── __init__.py
│   │   │   │   ├── market_cap_strategy.py
│   │   │   │   └── volume_profile_strategy.py
│   │   │   └── ml/                   # 机器学习策略
│   │   │       ├── __init__.py
│   │   │       ├── xgboost_strategy.py
│   │   │       └── lstm_strategy.py
│   │   │
│   │   ├── models/                   # 数据模型
│   │   │   ├── __init__.py
│   │   │   ├── strategy.py
│   │   │   ├── coin_score.py
│   │   │   ├── selection_result.py
│   │   │   └── backtest_result.py
│   │   │
│   │   ├── services/                 # 服务层
│   │   │   ├── __init__.py
│   │   │   ├── selection_service.py
│   │   │   ├── strategy_service.py
│   │   │   ├── backtest_service.py
│   │   │   └── market_data_client.py # 行情数据客户端
│   │   │
│   │   └── utils/                    # 工具函数
│   │       ├── __init__.py
│   │       ├── indicators.py         # 技术指标计算
│   │       ├── data_utils.py
│   │       ├── math_utils.py
│   │       └── config_loader.py
│   │
│   └── tests/                        # 服务专用测试
│       ├── test_strategy_engine.py
│       ├── test_strategies.py
│       └── test_backtest_engine.py
│
├── web-frontend/                     # Web前端 (React + TypeScript + MUI)
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── Dockerfile
│   │
│   ├── public/                       # 静态资源
│   │   ├── index.html
│   │   ├── favicon.ico
│   │   └── assets/
│   │
│   ├── src/                          # 源代码
│   │   ├── main.tsx                  # 应用入口
│   │   ├── App.tsx                   # 根组件
│   │   ├── index.css                 # 全局样式
│   │   │
│   │   ├── components/               # 通用组件
│   │   │   ├── common/               # 基础组件
│   │   │   │   ├── Button.tsx
│   │   │   │   ├── Input.tsx
│   │   │   │   ├── Modal.tsx
│   │   │   │   ├── Table.tsx
│   │   │   │   └── Loading.tsx
│   │   │   ├── charts/               # 图表组件
│   │   │   │   ├── CandlestickChart.tsx
│   │   │   │   ├── LineChart.tsx
│   │   │   │   └── VolumeChart.tsx
│   │   │   └── layout/               # 布局组件
│   │   │       ├── Header.tsx
│   │   │       ├── Sidebar.tsx
│   │   │       └── Footer.tsx
│   │   │
│   │   ├── pages/                    # 页面组件
│   │   │   ├── Dashboard/            # 仪表板
│   │   │   │   ├── index.tsx
│   │   │   │   ├── MarketOverview.tsx
│   │   │   │   └── RecentSelections.tsx
│   │   │   ├── MarketData/           # 行情数据
│   │   │   │   ├── index.tsx
│   │   │   │   ├── TickerList.tsx
│   │   │   │   └── CoinDetail.tsx
│   │   │   ├── CoinSelection/        # 选币系统
│   │   │   │   ├── index.tsx
│   │   │   │   ├── StrategyConfig.tsx
│   │   │   │   ├── SelectionResults.tsx
│   │   │   │   └── Backtest.tsx
│   │   │   └── Settings/             # 设置页面
│   │   │       ├── index.tsx
│   │   │       ├── ExchangeConfig.tsx
│   │   │       └── UserProfile.tsx
│   │   │
│   │   ├── hooks/                    # 自定义Hooks
│   │   │   ├── useWebSocket.ts
│   │   │   ├── useMarketData.ts
│   │   │   ├── useCoinSelection.ts
│   │   │   └── useAuth.ts
│   │   │
│   │   ├── services/                 # API服务
│   │   │   ├── api.ts                # API基础配置
│   │   │   ├── marketDataApi.ts
│   │   │   ├── coinSelectionApi.ts
│   │   │   └── authApi.ts
│   │   │
│   │   ├── store/                    # 状态管理
│   │   │   ├── index.ts
│   │   │   ├── slices/
│   │   │   │   ├── authSlice.ts
│   │   │   │   ├── marketDataSlice.ts
│   │   │   │   └── coinSelectionSlice.ts
│   │   │   └── middleware/
│   │   │       └── websocketMiddleware.ts
│   │   │
│   │   ├── types/                    # TypeScript类型定义
│   │   │   ├── api.ts
│   │   │   ├── market.ts
│   │   │   ├── strategy.ts
│   │   │   └── user.ts
│   │   │
│   │   └── utils/                    # 工具函数
│   │       ├── formatters.ts
│   │       ├── validators.ts
│   │       ├── constants.ts
│   │       └── helpers.ts
│   │
│   └── tests/                        # 前端测试
│       ├── components/
│       ├── pages/
│       └── utils/
│
├── shared/                           # 共享代码
│   ├── __init__.py
│   ├── models/                       # 共享数据模型
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── market_data.py
│   │   └── strategy.py
│   ├── utils/                        # 共享工具
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── redis_client.py
│   │   ├── kafka_client.py
│   │   └── logger.py
│   └── constants/                    # 共享常量
│       ├── __init__.py
│       ├── exchanges.py
│       └── timeframes.py
│
└── k8s/                              # Kubernetes配置
    ├── namespace.yaml
    ├── configmaps/
    │   ├── market-data-config.yaml
    │   └── coin-selection-config.yaml
    ├── secrets/
    │   ├── database-secret.yaml
    │   └── api-keys-secret.yaml
    ├── deployments/
    │   ├── market-data-deployment.yaml
    │   ├── coin-selection-deployment.yaml
    │   └── web-frontend-deployment.yaml
    ├── services/
    │   ├── market-data-service.yaml
    │   ├── coin-selection-service.yaml
    │   └── web-frontend-service.yaml
    ├── ingress/
    │   └── ingress.yaml
    └── monitoring/
        ├── prometheus-deployment.yaml
        └── grafana-deployment.yaml
```

## 关键文件说明

### 1. 配置文件

#### `.env.example`
```bash
# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/quant_system
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_influxdb_token
INFLUXDB_ORG=quant_org
INFLUXDB_BUCKET=market_data

# Redis配置
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_redis_password

# Kafka配置
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PREFIX=quant_

# API配置
API_SECRET_KEY=your_secret_key_here
API_ACCESS_TOKEN_EXPIRE_MINUTES=30

# 交易所API密钥（加密存储）
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret_key

# 监控配置
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000

# 日志配置
LOG_LEVEL=INFO
LOG_FILE_PATH=./logs/

# 开发环境配置
DEBUG=true
ENVIRONMENT=development
```

#### `Makefile`
```makefile
.PHONY: help install dev test build deploy clean

# 默认目标
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Start development environment"
	@echo "  test        - Run tests"
	@echo "  build       - Build Docker images"
	@echo "  deploy      - Deploy to production"
	@echo "  clean       - Clean up resources"

# 安装依赖
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd web-frontend && npm install

# 启动开发环境
dev:
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "Development environment started at http://localhost:3000"

# 运行测试
test:
	@echo "Running Python tests..."
	pytest tests/ -v
	@echo "Running frontend tests..."
	cd web-frontend && npm test

# 构建Docker镜像
build:
	@echo "Building Docker images..."
	docker-compose build

# 部署到生产环境
deploy:
	@echo "Deploying to production..."
	./scripts/deployment/deploy-prod.sh

# 清理资源
clean:
	@echo "Cleaning up..."
	docker-compose down -v
	docker system prune -f

# 数据库迁移
migrate:
	@echo "Running database migrations..."
	alembic upgrade head

# 生成API文档
docs:
	@echo "Generating API documentation..."
	cd market-data-server && python -m app.generate_docs
	cd coin-selection-system && python -m app.generate_docs
```

### 2. 开发工具配置

#### `.gitignore`
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Environment Variables
.env
.env.local
.env.development
.env.production

# Database
*.db
*.sqlite3

# Logs
logs/
*.log

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Build outputs
build/
dist/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Kubernetes
kustomization.yaml

# Monitoring
prometheus_data/
grafana_data/

# Backup files
*.bak
*.backup

# Temporary files
tmp/
temp/
```

### 3. 开发规范

#### Python代码规范
- 使用 Black 进行代码格式化
- 使用 isort 进行导入排序
- 使用 flake8 进行代码检查
- 使用 mypy 进行类型检查
- 遵循 PEP 8 编码规范

#### TypeScript代码规范
- 使用 ESLint 进行代码检查
- 使用 Prettier 进行代码格式化
- 严格的 TypeScript 配置
- 组件使用函数式组件 + Hooks

#### Git提交规范
```
feat: 新功能
fix: 修复bug
docs: 文档更新
style: 代码格式调整
refactor: 代码重构
test: 测试相关
chore: 构建过程或辅助工具的变动
```

### 4. 部署结构

#### 开发环境
- 本地Docker Compose部署
- 热重载开发
- 调试模式启用
- 详细日志输出

#### 测试环境
- 模拟生产环境配置
- 自动化测试执行
- 性能测试
- 安全测试

#### 生产环境
- Kubernetes集群部署
- 高可用配置
- 自动扩缩容
- 监控告警
- 备份恢复

---

*此项目结构设计遵循微服务架构原则，确保各模块独立开发、测试和部署。*