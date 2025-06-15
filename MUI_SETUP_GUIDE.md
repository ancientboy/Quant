# Material-UI (MUI) 配置指南

## 概述

本项目前端采用 **Material-UI (MUI)** 作为 UI 组件库，提供现代化、一致性的用户界面设计。MUI 基于 Google 的 Material Design 设计语言，特别适合数据密集型的金融交易应用。

## 技术栈

- **React** 18+
- **TypeScript** 5+
- **Material-UI (MUI)** 5+
- **Emotion** (CSS-in-JS)
- **Vite** (构建工具)

## 安装依赖

### 核心依赖
```bash
npm install @mui/material @emotion/react @emotion/styled
```

### 图标库
```bash
npm install @mui/icons-material
```

### 日期选择器 (可选)
```bash
npm install @mui/x-date-pickers dayjs
```

### 数据表格 (推荐用于交易数据展示)
```bash
npm install @mui/x-data-grid
```

### 图表库 (用于K线图等)
```bash
npm install @mui/x-charts
# 或者使用更专业的图表库
npm install recharts
```

## 项目配置

### 1. 主题配置 (`src/theme/index.ts`)

```typescript
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';

// 创建自定义主题
const theme = createTheme({
  palette: {
    mode: 'dark', // 交易界面通常使用暗色主题
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#0a0e27',
      paper: '#1e2139',
    },
    success: {
      main: '#00c853', // 涨幅颜色
    },
    error: {
      main: '#f44336', // 跌幅颜色
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
  },
  components: {
    // 自定义组件样式
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
        },
      },
    },
  },
});

export default theme;
```

### 2. 应用入口配置 (`src/main.tsx`)

```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import App from './App';
import theme from './theme';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
);
```

## 核心组件示例

### 1. 交易仪表板布局

```typescript
import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  AppBar,
  Toolbar,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  Dashboard,
  TrendingUp,
  Settings,
  AccountBalance,
} from '@mui/icons-material';

const TradingDashboard: React.FC = () => {
  return (
    <Box sx={{ display: 'flex' }}>
      {/* 侧边栏 */}
      <Drawer
        variant="permanent"
        sx={{
          width: 240,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 240,
            boxSizing: 'border-box',
          },
        }}
      >
        <List>
          <ListItem button>
            <ListItemIcon>
              <Dashboard />
            </ListItemIcon>
            <ListItemText primary="仪表板" />
          </ListItem>
          <ListItem button>
            <ListItemIcon>
              <TrendingUp />
            </ListItemIcon>
            <ListItemText primary="行情数据" />
          </ListItem>
          <ListItem button>
            <ListItemIcon>
              <AccountBalance />
            </ListItemIcon>
            <ListItemText primary="选币系统" />
          </ListItem>
        </List>
      </Drawer>

      {/* 主内容区 */}
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <AppBar position="static" sx={{ mb: 3 }}>
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              量化交易系统
            </Typography>
          </Toolbar>
        </AppBar>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  总资产
                </Typography>
                <Typography variant="h5" component="div">
                  $125,430.50
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          {/* 更多卡片... */}
        </Grid>
      </Box>
    </Box>
  );
};

export default TradingDashboard;
```

### 2. 数据表格组件 (用于显示币种列表)

```typescript
import React from 'react';
import {
  DataGrid,
  GridColDef,
  GridValueGetterParams,
} from '@mui/x-data-grid';
import { Chip, Box } from '@mui/material';

interface CoinData {
  id: string;
  symbol: string;
  name: string;
  price: number;
  change24h: number;
  volume24h: number;
  marketCap: number;
}

const CoinListTable: React.FC<{ data: CoinData[] }> = ({ data }) => {
  const columns: GridColDef[] = [
    { field: 'symbol', headerName: '币种', width: 100 },
    { field: 'name', headerName: '名称', width: 150 },
    {
      field: 'price',
      headerName: '价格',
      width: 120,
      type: 'number',
      valueFormatter: (params) => `$${params.value.toFixed(4)}`,
    },
    {
      field: 'change24h',
      headerName: '24h涨跌',
      width: 120,
      renderCell: (params) => (
        <Chip
          label={`${params.value > 0 ? '+' : ''}${params.value.toFixed(2)}%`}
          color={params.value > 0 ? 'success' : 'error'}
          size="small"
        />
      ),
    },
    {
      field: 'volume24h',
      headerName: '24h成交量',
      width: 150,
      type: 'number',
      valueFormatter: (params) => `$${(params.value / 1000000).toFixed(2)}M`,
    },
  ];

  return (
    <Box sx={{ height: 600, width: '100%' }}>
      <DataGrid
        rows={data}
        columns={columns}
        pageSize={25}
        rowsPerPageOptions={[25, 50, 100]}
        checkboxSelection
        disableSelectionOnClick
        sx={{
          '& .MuiDataGrid-cell': {
            borderColor: 'rgba(255, 255, 255, 0.12)',
          },
          '& .MuiDataGrid-columnHeaders': {
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
          },
        }}
      />
    </Box>
  );
};

export default CoinListTable;
```

### 3. 策略配置表单

```typescript
import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  TextField,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Button,
  Grid,
  Typography,
  Box,
} from '@mui/material';

const StrategyConfigForm: React.FC = () => {
  const [config, setConfig] = useState({
    strategyName: '',
    riskLevel: 5,
    maxPositions: 10,
    enableStopLoss: true,
    stopLossPercent: 5,
    timeframe: '1h',
  });

  return (
    <Card>
      <CardHeader title="策略配置" />
      <CardContent>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="策略名称"
              value={config.strategyName}
              onChange={(e) => setConfig({ ...config, strategyName: e.target.value })}
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>时间周期</InputLabel>
              <Select
                value={config.timeframe}
                label="时间周期"
                onChange={(e) => setConfig({ ...config, timeframe: e.target.value })}
              >
                <MenuItem value="5m">5分钟</MenuItem>
                <MenuItem value="15m">15分钟</MenuItem>
                <MenuItem value="1h">1小时</MenuItem>
                <MenuItem value="4h">4小时</MenuItem>
                <MenuItem value="1d">1天</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12}>
            <Typography gutterBottom>风险等级: {config.riskLevel}</Typography>
            <Slider
              value={config.riskLevel}
              onChange={(_, value) => setConfig({ ...config, riskLevel: value as number })}
              min={1}
              max={10}
              marks
              valueLabelDisplay="auto"
            />
          </Grid>

          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              type="number"
              label="最大持仓数量"
              value={config.maxPositions}
              onChange={(e) => setConfig({ ...config, maxPositions: parseInt(e.target.value) })}
            />
          </Grid>

          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={config.enableStopLoss}
                  onChange={(e) => setConfig({ ...config, enableStopLoss: e.target.checked })}
                />
              }
              label="启用止损"
            />
          </Grid>

          {config.enableStopLoss && (
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="止损百分比 (%)"
                value={config.stopLossPercent}
                onChange={(e) => setConfig({ ...config, stopLossPercent: parseFloat(e.target.value) })}
              />
            </Grid>
          )}

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
              <Button variant="outlined">重置</Button>
              <Button variant="contained">保存配置</Button>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default StrategyConfigForm;
```

## 开发规范

### 1. 组件命名
- 使用 PascalCase 命名组件文件
- 组件名称应该描述其功能
- 例如: `CoinListTable.tsx`, `StrategyConfigForm.tsx`

### 2. 样式规范
- 优先使用 MUI 的 `sx` prop 进行样式定制
- 复杂样式使用 `styled` 组件
- 避免内联样式，保持代码整洁

### 3. 主题使用
- 所有颜色值从主题中获取
- 使用主题的断点进行响应式设计
- 保持设计一致性

### 4. 性能优化
- 使用 `React.memo` 优化重渲染
- 大数据表格使用虚拟滚动
- 图表组件按需加载

## 推荐的第三方库

### 图表库
```bash
# 专业的金融图表库
npm install lightweight-charts

# 通用图表库
npm install recharts
```

### 状态管理
```bash
# Redux Toolkit (推荐)
npm install @reduxjs/toolkit react-redux

# 或者 Zustand (轻量级)
npm install zustand
```

### 表单处理
```bash
npm install react-hook-form @hookform/resolvers yup
```

### WebSocket
```bash
npm install socket.io-client
```

## 部署配置

### Dockerfile
```dockerfile
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 总结

Material-UI 为我们的量化交易系统提供了:

1. **专业的设计语言**: Material Design 确保界面现代化和一致性
2. **丰富的组件库**: 数据表格、图表、表单等开箱即用
3. **优秀的 TypeScript 支持**: 类型安全，开发体验好
4. **主题定制能力**: 支持暗色模式，适合交易界面
5. **性能优化**: 虚拟滚动、懒加载等特性
6. **活跃的社区**: 文档完善，问题解决快

通过合理使用 MUI，我们可以快速构建出专业、美观、高性能的量化交易前端界面。