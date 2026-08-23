#!/usr/bin/env bash
# Ставит диктофон стакана как отдельную службу systemd.
#
# ВАЖНО: этот проект намеренно не пересекается с trading-bot. Свой каталог,
# своё окружение, своя служба, свой каталог данных. Установщик не трогает
# ни файлы бота, ни его службы — прогон в это время идёт как шёл.
#
# Запуск:  [СЕРВЕР]  cd /root/orderbook-recorder && bash install_service.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PROJECT_DIR}/venv/bin/python"
SERVICE=/etc/systemd/system/orderbook-recorder.service

echo "==> Проект: ${PROJECT_DIR}"

if [[ "${PROJECT_DIR}" == *"trading-bot"* ]]; then
    echo "ОШИБКА: диктофон лежит внутри trading-bot. Он должен стоять отдельно."
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "==> Создаю окружение"
    python3 -m venv "${PROJECT_DIR}/venv"
    "${PROJECT_DIR}/venv/bin/pip" install -q --upgrade pip
    "${PROJECT_DIR}/venv/bin/pip" install -q -r "${PROJECT_DIR}/requirements.txt"
fi

echo "==> Проверяю библиотеки"
"$PYTHON" -c "import websockets, pyarrow, requests, certifi" || {
    echo "ОШИБКА: не хватает библиотек. Выполните:"
    echo "  ${PROJECT_DIR}/venv/bin/pip install -r ${PROJECT_DIR}/requirements.txt"
    exit 1
}

echo "==> Проверяю место на диске"
df -h "${PROJECT_DIR}" | tail -1

echo "==> Короткий прогон перед установкой (60 с)"
"$PYTHON" "${PROJECT_DIR}/tools/smoke.py" 60 || {
    echo
    echo "ОШИБКА: проверка не прошла — служба не поставлена. Разберитесь выше."
    exit 1
}

echo "==> Пишу ${SERVICE}"
cat > "$SERVICE" <<EOF
[Unit]
Description=MEXC order book recorder (research data capture)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PYTHON} ${PROJECT_DIR}/run.py

# Перезапуск при любом падении. Пауза короче, чем у бота: пропущенные секунды
# потока не восстановить, а торговых последствий у перезапуска нет.
Restart=always
RestartSec=15

StandardOutput=append:${PROJECT_DIR}/recorder.log
StandardError=append:${PROJECT_DIR}/recorder.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable orderbook-recorder >/dev/null 2>&1
systemctl restart orderbook-recorder

sleep 10
STATE="$(systemctl is-active orderbook-recorder || true)"
echo
echo "============================================================"
if [[ "$STATE" == "active" ]]; then
    echo "  ГОТОВО — диктофон пишет"
    echo "============================================================"
    echo
    echo "  Что где:"
    echo "    данные   ${PROJECT_DIR}/data/ГГГГ-ММ-ДД/events_*.parquet"
    echo "    журнал   ${PROJECT_DIR}/recorder.log"
    echo
    echo "  Команды:"
    echo "    отчёт        ${PYTHON} ${PROJECT_DIR}/tools/report.py"
    echo "    журнал       journalctl -u orderbook-recorder -f"
    echo "    остановить   systemctl stop orderbook-recorder"
    echo
    echo "  Бот не тронут:"
    systemctl is-active trading-bot 2>/dev/null | sed 's/^/    trading-bot: /' || true
else
    echo "  НЕ ЗАПУСТИЛАСЬ — состояние: ${STATE}"
    echo "============================================================"
    journalctl -u orderbook-recorder -n 25 --no-pager
    exit 1
fi
