#!/usr/bin/env bash
# Ставит страницу со стаканом отдельной службой.
#
# Почему отдельно от install_service.sh: диктофон — главное, он уже работает,
# и трогать его ради страницы не нужно. Эта установка ничего в нём не меняет.
#
# Порт 80 занят страницей состояния бота, поэтому берём 8080.
#
# Запуск:  bash install_dashboard.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PROJECT_DIR}/venv/bin/python"
PORT="${OBR_DASH_PORT:-8080}"
SERVICE=/etc/systemd/system/orderbook-dashboard.service

echo "==> Проект: ${PROJECT_DIR}, порт ${PORT}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ОШИБКА: нет ${PYTHON}. Сначала поставьте диктофон: bash install_service.sh"
    exit 1
fi

echo "==> Пишу ${SERVICE}"
cat > "$SERVICE" <<UNIT
[Unit]
Description=MEXC order book dashboard (read-only web page)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${PROJECT_DIR}
Environment=OBR_DASH_PORT=${PORT}
ExecStart=${PYTHON} ${PROJECT_DIR}/dashboard.py
Restart=always
RestartSec=15
StandardOutput=append:${PROJECT_DIR}/dashboard.log
StandardError=append:${PROJECT_DIR}/dashboard.log

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable orderbook-dashboard >/dev/null 2>&1
systemctl restart orderbook-dashboard

# Порт наружу: без этого страница не откроется, а молча пустая страница
# отлаживается дольше, чем печатается эта строка.
if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "^Status: active"; then
    echo "==> Открываю порт ${PORT} в ufw"
    ufw allow "${PORT}/tcp" >/dev/null || true
else
    echo "==> ufw не активен — порт и так открыт"
fi

sleep 6
STATE="$(systemctl is-active orderbook-dashboard || true)"
echo
echo "============================================================"
if [[ "$STATE" == "active" ]]; then
    IP="$(curl -4 -s --max-time 5 ifconfig.me || echo '<IP-сервера>')"
    TOKEN="$(cat "${PROJECT_DIR}/dashboard_token.txt" 2>/dev/null || echo '<токен>')"
    echo "  ГОТОВО — страница работает"
    echo "============================================================"
    echo
    echo "    http://${IP}:${PORT}/${TOKEN}"
    echo
    echo "  Без токена в адресе отдаётся 404. Связь без шифрования,"
    echo "  поэтому ссылку никому не пересылайте и в закладках держите"
    echo "  только у себя."
    echo
    echo "    журнал      tail -f ${PROJECT_DIR}/dashboard.log"
    echo "    остановить  systemctl stop orderbook-dashboard"
    echo
    echo "  Диктофон не тронут:"
    systemctl is-active orderbook-recorder 2>/dev/null | sed 's/^/    orderbook-recorder: /' || true
    systemctl is-active trading-bot 2>/dev/null | sed 's/^/    trading-bot: /' || true
else
    echo "  НЕ ЗАПУСТИЛАСЬ — состояние: ${STATE}"
    echo "============================================================"
    tail -n 25 "${PROJECT_DIR}/dashboard.log" 2>/dev/null || true
    exit 1
fi
