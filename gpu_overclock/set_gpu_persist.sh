#!/bin/bash

echo "🔧 Starting freq offset..."
nvidia-settings -a "GPUGraphicsClockOffset[4]=300"

echo "🔧 Starting NVIDIA Persistence Daemon..."

# 尝试启动 nvidia-persistenced
sudo systemctl start nvidia-persistenced

# 检查是否启动成功
STATUS=$(systemctl is-active nvidia-persistenced)
if [ "$STATUS" = "active" ]; then
    echo "✅ nvidia-persistenced is running."
else
    echo "❌ Failed to start nvidia-persistenced. Current status: $STATUS"
    exit 1
fi

# 检查是否能读取 GPU 状态
echo "📊 Checking GPU frequency state..."
nvidia-smi --query-gpu=name,persistence_mode,clocks.gr,clocks.mem --format=csv

echo "✅ Done."
