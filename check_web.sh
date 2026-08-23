#!/usr/bin/env bash
# Почему не открывается страница: проверяет всю цепочку по порядку.
#
# Порядок важен — каждый следующий шаг имеет смысл только если предыдущий
# прошёл. Скрипт печатает вывод по-русски и в конце говорит, что делать.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${OBR_DASH_PORT:-8080}"
TOKEN="$(cat "${PROJECT_DIR}/dashboard_token.txt" 2>/dev/null)"
OK=1

echo "============================================================"
echo "  ПРОВЕРКА ДОСТУПА К СТРАНИЦЕ (порт ${PORT})"
echo "============================================================"

echo
echo "1. Служба"
STATE="$(systemctl is-active orderbook-dashboard 2>/dev/null)"
if [[ "$STATE" == "active" ]]; then
    echo "   OK   orderbook-dashboard работает"
else
    echo "   СБОЙ состояние: ${STATE:-нет службы}"
    echo "        последние строки журнала:"
    tail -n 12 "${PROJECT_DIR}/dashboard.log" 2>/dev/null | sed 's/^/        /'
    OK=0
fi

echo
echo "2. Слушает ли порт"
LISTEN="$(ss -tln 2>/dev/null | grep ":${PORT} ")"
if [[ -n "$LISTEN" ]]; then
    echo "   OK   $(echo "$LISTEN" | awk '{print $4}' | head -1)"
    if ! echo "$LISTEN" | grep -qE '0\.0\.0\.0|\*|\[::\]'; then
        echo "   ВНИМАНИЕ: слушает только локально, снаружи не откроется"
        OK=0
    fi
else
    echo "   СБОЙ никто не слушает порт ${PORT}"
    OK=0
fi

echo
echo "3. Отвечает ли изнутри сервера"
if [[ -z "$TOKEN" ]]; then
    echo "   СБОЙ нет dashboard_token.txt"
    OK=0
else
    CODE="$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:${PORT}/${TOKEN}")"
    if [[ "$CODE" == "200" ]]; then
        echo "   OK   страница отдаётся (HTTP 200)"
    else
        echo "   СБОЙ локальный запрос вернул ${CODE}"
        OK=0
    fi
fi

echo
echo "4. Локальный межсетевой экран"
if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "^Status: active"; then
    if ufw status | grep -q "${PORT}"; then
        echo "   OK   ufw активен, порт ${PORT} разрешён"
    else
        echo "   СБОЙ ufw активен и порт ${PORT} не открыт. Выполните:"
        echo "        ufw allow ${PORT}/tcp"
        OK=0
    fi
else
    echo "   OK   ufw не активен, локально ничего не блокирует"
fi

IP="$(curl -4 -s --max-time 5 ifconfig.me 2>/dev/null || echo '<IP>')"
echo
echo "============================================================"
if [[ "$OK" == "1" ]]; then
    echo "  На сервере всё в порядке. Адрес:"
    echo
    echo "    http://${IP}:${PORT}/${TOKEN}"
    echo
    echo "  Если он всё равно не открывается, остаются две причины,"
    echo "  и обе вне сервера:"
    echo
    echo "  1) ОБЛАЧНЫЙ ФАЙРВОЛ HETZNER. Он настраивается в панели, а не"
    echo "     на сервере, и ufw про него ничего не знает. Обычно там"
    echo "     разрешены только 22, 80 и 443 — тогда 8080 закрыт."
    echo "     Панель Hetzner → Firewalls → правило: inbound TCP ${PORT}."
    echo
    echo "  2) БРАУЗЕР ПОДСТАВЛЯЕТ HTTPS. Адрес именно http://, буквы s"
    echo "     нет. Наберите его целиком, не из подсказки строки поиска."
    echo
    echo "  Проверить снаружи, не открывая браузер, можно с телефона по"
    echo "  мобильному интернету — если и там глухо, это файрвол."
else
    echo "  Есть проблемы на сервере — смотрите СБОЙ выше."
fi
echo "============================================================"
