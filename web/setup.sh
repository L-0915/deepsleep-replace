#!/bin/bash
set -e
cd /root/dslm/deepsleep/web

echo "=== DeepSleep Web 前端安装 ==="
echo ""

echo "[1/2] 安装依赖..."
npm install

echo ""
echo "[2/2] 构建项目..."
npm run build

echo ""
echo "=== 完成! ==="
echo "静态文件已输出到: /root/dslm/deepsleep/static/"
echo ""
echo "开发模式运行: cd /root/dslm/deepsleep/web && npm run dev"
