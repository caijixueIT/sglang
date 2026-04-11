#!/bin/bash
# install_local.sh - 安装本地 sglang 代码库 (开发模式)
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$SCRIPT_DIR/python"

echo "=========================================="
echo "安装本地 SGLang 代码库"
echo "路径: $SGLANG_DIR"
echo "=========================================="

# 检查 python 目录是否存在
if [ ! -d "$SGLANG_DIR" ]; then
    echo "[ERROR] 找不到 python 目录: $SGLANG_DIR"
    exit 1
fi

# 检查 pyproject.toml 是否存在
if [ ! -f "$SGLANG_DIR/pyproject.toml" ]; then
    echo "[ERROR] 找不到 pyproject.toml: $SGLANG_DIR/pyproject.toml"
    exit 1
fi

# 临时注释掉尚未发布到 PyPI 的依赖
PYPROJECT="$SGLANG_DIR/pyproject.toml"
if grep -q '"smg-grpc-proto' "$PYPROJECT" && ! grep -q '# "smg-grpc-proto' "$PYPROJECT"; then
    echo "[INFO] 临时注释 smg-grpc-proto 依赖 (尚未发布到 PyPI)..."
    sed -i 's/"smg-grpc-proto/# "smg-grpc-proto/' "$PYPROJECT"
fi

# 安装 (开发模式)
echo "[INFO] 开始安装 sglang (开发模式)..."
cd "$SCRIPT_DIR"
pip install -e "python" --no-build-isolation

# 验证安装
echo ""
echo "=========================================="
echo "安装完成！验证中..."
echo "=========================================="
python -c "
import sglang
print(f'sglang 版本: {sglang.__version__}')
print(f'安装路径: {sglang.__file__}')
"

echo ""
echo "✅ 本地 sglang 已安装成功！"
echo "   修改代码后重启服务即可生效，无需重新安装。"
