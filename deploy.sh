#!/bin/bash

cd /root/Mariah-s-Dashboard || exit
echo "🔁 Pulling latest from GitHub..."
git pull origin main

echo "🚀 Restarting dashboard service..."
systemctl restart mariah-dashboard
