#!/bin/bash

echo "🚀 Bắt đầu quy trình đồng bộ hóa với GitHub..."

echo "1️⃣ Chuyển về nhánh chính (main)..."
git checkout main

echo "2️⃣ Đang tải dữ liệu mới nhất từ origin..."
git fetch origin

echo "3️⃣ Đang ép đồng bộ tuyệt đối (Hard Reset)..."
git reset --hard origin/main

echo "✨ XONG!"