#!/bin/bash

# 设置遇到错误时停止执行
set -e

# 运行第一个 Python 脚本
echo "Running RLagent.py"
python3 RLagent.py

# 运行第二个 Python 脚本
echo "Running ACagent.py"
python3 ACagent.py

# 运行第三个 Python 脚本
echo "Running A2Cagent.py"
python3 A2Cagent.py

# 运行第四个 Python 脚本
echo "Running PPOagent.py"
python3 PPOagent.py

echo "All scripts executed successfully!"
