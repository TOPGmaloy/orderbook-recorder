#!/usr/bin/env bash
# Ставит карту стоимости входа как отдельную службу systemd.
#
# Это второй режим записи, рядом с диктофоном и независимо от него: свой
# процесс, свой каталог данных (data_cost), своя служба. Диктофон при установке
# не трогается и продолжает писать как писал. Ключей API здесь нет, как и там.
#
# Запуск:  [СЕРВЕР]  cd /root/orderbook-recorder && bash install_costmap.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PROJECT_DIR}/venv/bin/python"
SERVICE=/etc/systemd/system/orderbook-costmap.service

echo "==> Проект: ${PROJECT_DIR}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ОШИБКА: нет окружения. Сначала поставьте диктофон:"
    echo "  cd ${PROJECT_DIR} && bash install_service.sh"
    exit 1
fi

echo "==> Проверяю библиотеки"
"$PYTHON" -c "import pyarrow, requests" || {
    echo "ОШИБКА: не хватает библиотек. Выполните:"
    echo "  ${PROJECT_DIR}/venv/bin/pip install -r ${PROJECT_DIR}/requirements.txt"
    exit 1
}

echo "==> Пробный обход (один проход по универсу, без записи службы)"
"$PYTHON" - <<'PY'
import sys, time
sys.path.insert(0, ".")
from recorder.costmap import universe, probe
pairs = universe()
print(f"    универс: {len(pairs)} пар выше порога оборота")
if not pairs:
    raise SystemExit("    биржа не отдала список контрактов — установка прервана")
ok = 0
for symbol, size, turnover in pairs[:5]:
    if probe(symbol, size, turnover):
        ok += 1
    time.sleep(0.25)
print(f"    пробные снимки: {ok} из 5")
if ok < 3:
    raise SystemExit("    книга не читается — установка прервана")
PY

echo "==> Пишу юнит ${SERVICE}"
cat > "$SERVICE" <<UNIT
[Unit]
Description=MEXC entry cost map (momentum bot universe)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PYTHON} ${PROJECT_DIR}/run_costmap.py
Restart=always
RestartSec=30
StandardOutput=append:${PROJECT_DIR}/costmap.log
StandardError=append:${PROJECT_DIR}/costmap.log

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now orderbook-costmap
sleep 5
systemctl --no-pager status orderbook-costmap | head -12

echo
echo "==> Готово. Журнал:"
echo "    tail -f ${PROJECT_DIR}/costmap.log"
echo "==> Отчёт (через час-другой, когда накопится):"
echo "    cd ${PROJECT_DIR} && ./costmap"
