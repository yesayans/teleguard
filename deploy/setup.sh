#!/usr/bin/env bash
set -e

echo "=================================================="
echo "  TeleGuard AI: Automated Server Setup Script     "
echo "  Target Domain: teleguard.yesayan.net            "
echo "=================================================="

# Check if Docker is installed
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "[+] Docker and Docker Compose detected. Starting via Docker..."
    docker-compose up -d --build
    echo "[+] TeleGuard AI is live with automatic SSL on https://teleguard.yesayan.net"
    exit 0
fi

# Fallback: Native systemd + Nginx setup
echo "[+] Setting up native environment on Ubuntu/Debian..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv libsndfile1 ffmpeg nginx certbot python3-certbot-nginx

# Create directory
sudo mkdir -p /var/www/teleguard
sudo cp -r . /var/www/teleguard/
sudo chown -R www-data:www-data /var/www/teleguard

# Create venv
sudo -u www-data python3 -m venv /var/www/teleguard/venv
sudo -u www-data /var/www/teleguard/venv/bin/pip install --upgrade pip
sudo -u www-data /var/www/teleguard/venv/bin/pip install -r /var/www/teleguard/requirements.txt

# Install systemd service
sudo cp /var/www/teleguard/deploy/teleguard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now teleguard

# Configure Nginx
sudo cp /var/www/teleguard/deploy/nginx.conf /etc/nginx/sites-available/teleguard
sudo ln -sf /etc/nginx/sites-available/teleguard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Issue Let's Encrypt SSL
echo "[+] Obtaining SSL Certificate from Let's Encrypt for teleguard.yesayan.net..."
sudo certbot --nginx -d teleguard.yesayan.net --non-interactive --agree-tos --register-unsafely-without-email || true

echo "=================================================="
echo "  [SUCCESS] TeleGuard AI is active and running!   "
echo "  Visit: https://teleguard.yesayan.net            "
echo "=================================================="
