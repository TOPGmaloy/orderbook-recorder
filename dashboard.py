#!/usr/bin/env python3
"""Страница со стаканом и лентой — смотреть из браузера, а не из консоли.

Отдельная служба: своё подключение к публичному потоку MEXC, диктофону не
мешает, ничего не пишет и не торгует. Управлять записью со страницы нельзя —
она только показывает.

Адрес защищён токеном в пути, как страница состояния бота: без него отдаётся
404. Это HTTP без шифрования, поэтому токен идёт по сети открытым — для
публичных котировок и статуса записи это приемлемо, для чего-то большего эту
страницу использовать нельзя.

Запуск вручную:  python dashboard.py       (порт из OBR_DASH_PORT, по умолчанию 8080)
"""

import asyncio
import json
import os
import secrets
import shutil
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

from config import SYMBOLS, DATA_DIR
from recorder.book import OrderBook
from recorder.mexc_ws import MexcFeed, fetch_snapshot

ROOT = Path(__file__).resolve().parent
PORT = int(os.getenv("OBR_DASH_PORT", "8080"))
TOKEN_FILE = ROOT / "dashboard_token.txt"
OUT_DIR = ROOT / "out"          # сюда обёртки складывают выводы инструментов

STATE = {"symbols": {}, "status": {}}     # читается сервером, пишется циклом


def token():
    if TOKEN_FILE.exists():
        value = TOKEN_FILE.read_text().strip()
        if value:
            return value
    value = secrets.token_urlsafe(18)
    TOKEN_FILE.write_text(value)
    TOKEN_FILE.chmod(0o600)
    return value


def decimals(step):
    """Сколько знаков после запятой нужно, чтобы шаг цены был виден.

    Округлять по величине цены нельзя: у ETH цена 2472, но шаг 0.01, и при
    одном знаке соседние уровни стакана сливаются в один.
    """
    text = f"{float(step):.10f}".rstrip("0")
    return max(0, len(text.split(".")[1]) if "." in text else 0)


def contract_info():
    sizes = {"BTC_USDT": 0.0001, "ETH_USDT": 0.01, "SOL_USDT": 0.1,
             "XAU_USDT": 0.001, "HYPE_USDT": 0.1, "XRP_USDT": 1.0}
    digits = {"BTC_USDT": 1, "ETH_USDT": 2, "SOL_USDT": 2, "XAU_USDT": 2, "HYPE_USDT": 3, "XRP_USDT": 4}
    try:
        body = requests.get("https://contract.mexc.com/api/v1/contract/detail",
                            timeout=15).json()
        for row in body.get("data") or []:
            if row.get("symbol") in SYMBOLS:
                sizes[row["symbol"]] = float(row["contractSize"])
                digits[row["symbol"]] = decimals(row["priceUnit"])
    except Exception:
        pass
    return sizes, digits


# --- сбор состояния ---------------------------------------------------------

class Live:
    def __init__(self):
        self.books = {s: OrderBook(s) for s in SYMBOLS}
        self.tape = {s: deque(maxlen=600) for s in SYMBOLS}
        self.sizes, self.digits = contract_info()
        self.started = time.time()

    async def on_message(self, message, ts_us):
        symbol = message.get("symbol")
        if symbol not in self.books:
            return
        channel = message.get("channel", "")
        data = message.get("data")
        if channel == "push.depth":
            if self.books[symbol].apply_delta(data) == "gap":
                await self.resync(symbol)
        elif channel == "push.deal":
            for deal in (data if isinstance(data, list) else [data]):
                try:
                    self.tape[symbol].append((time.time(), float(deal["p"]),
                                              float(deal["v"]), int(deal["T"])))
                except (KeyError, TypeError, ValueError):
                    continue

    async def resync(self, symbol=None):
        for s in ([symbol] if symbol else SYMBOLS):
            data, _ = await asyncio.to_thread(fetch_snapshot, s)
            if data:
                self.books[s].apply_snapshot(data)

    def snapshot(self, levels=12):
        out = {}
        now = time.time()
        for s in SYMBOLS:
            book = self.books[s]
            b, a = book.best()
            if not b or not a:
                out[s] = {"ready": False}
                continue
            mid = (b[0] + a[0]) / 2
            unit = self.sizes.get(s, 1.0) * mid
            bids = [[p, book.bids[p] * unit] for p in
                    sorted(book.bids, reverse=True)[:levels]]
            asks = [[p, book.asks[p] * unit] for p in sorted(book.asks)[:levels]]
            recent = [t for t in self.tape[s] if now - t[0] <= 5]
            qb, qa = b[1], a[1]
            out[s] = {
                "ready": True, "mid": mid,
                "spread_bp": (a[0] - b[0]) / mid * 1e4,
                "imbalance": (qb - qa) / (qb + qa) if qb + qa else 0.0,
                "bids": bids, "asks": asks,
                "buy5": sum(t[2] for t in recent if t[3] == 1) * unit,
                "sell5": sum(t[2] for t in recent if t[3] == 2) * unit,
                "count5": len(recent),
                "last": [[p, sd] for _, p, _, sd in list(self.tape[s])[-14:]],
                "digits": self.digits.get(s, 2),
            }
        return out

    def status(self):
        line = ""
        log = ROOT / "recorder.log"
        if log.exists():
            try:
                tail = log.read_text(errors="replace").strip().splitlines()
                for row in reversed(tail[-80:]):
                    if "аптайм" in row:
                        line = row.split("recorder: ", 1)[-1]
                        break
            except Exception:
                pass
        free = shutil.disk_usage(DATA_DIR).free / 1e9 if DATA_DIR.exists() else 0
        size = sum(f.stat().st_size for f in DATA_DIR.rglob("*.parquet")) / 1e6 \
            if DATA_DIR.exists() else 0
        return {"recorder": line, "free_gb": round(free, 1),
                "data_mb": round(size), "watch_uptime": int(time.time() - self.started)}


async def collector():
    live = Live()
    feed = MexcFeed(SYMBOLS, on_reconnect=live.resync)
    await live.resync()

    async def refresh():
        while True:
            STATE["symbols"] = live.snapshot()
            STATE["status"] = live.status()
            await asyncio.sleep(1)

    asyncio.create_task(refresh())
    await feed.run(live.on_message)


# --- страница ---------------------------------------------------------------

PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Стакан MEXC</title>
<style>
  :root {
    --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
    --muted:#898781; --line:rgba(11,11,11,.10);
    --buy:#2a78d6; --sell:#eb6834; color-scheme:light;
  }
  @media (prefers-color-scheme:dark){ :root{
    --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
    --muted:#898781; --line:rgba(255,255,255,.10);
    --buy:#3987e5; --sell:#d95926; color-scheme:dark; } }
  *{box-sizing:border-box}
  body{margin:0;background:var(--plane);color:var(--ink);
       font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:16px}
  .wrap{max-width:1180px;margin:0 auto}
  h1{font-size:18px;margin:0 0 4px;font-weight:600}
  .sub{color:var(--ink2);font-size:12.5px;margin:0 0 14px}
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
  .chip{background:var(--surface);border:1px solid var(--line);border-radius:6px;
        padding:4px 9px;font-size:12px;color:var(--ink2);
        font-variant-numeric:tabular-nums}
  .chip b{color:var(--ink);font-weight:600}
  .grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(310px,1fr))}
  .card{background:var(--surface);border:1px solid var(--line);
        border-radius:10px;padding:12px 14px}
  .card h2{font-size:13px;margin:0 0 2px;font-weight:600}
  .meta{font-size:11.5px;color:var(--muted);margin:0 0 10px;
        font-variant-numeric:tabular-nums}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  td{padding:1px 4px;font-size:12px;white-space:nowrap}
  td.p{color:var(--ink2);width:36%} td.n{text-align:right;width:22%}
  td.b{width:42%} .bx{height:9px;border-radius:2px}
  tr.touch td{border-top:1px solid var(--muted)}
  tr.touch td.p{color:var(--ink);font-weight:600}
  .tape{margin-top:10px;padding-top:9px;border-top:1px solid var(--line);
        font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums}
  .last{margin-top:4px;font-size:11.5px;word-spacing:4px}
  .up{color:var(--buy)} .dn{color:var(--sell)}
  .foot{margin-top:16px;font-size:12px;color:var(--muted)}
</style></head><body><div class="wrap">
<h1>Стакан MEXC</h1>
<p class="sub">Живой поток. Синим — биды и покупки по аску, оранжевым — аски и
продажи по биду. Объёмы в деньгах по номиналу. Обновление раз в секунду.</p>
<div class="chips" id="chips"></div>
<div class="grid" id="grid"></div>
<p class="foot" id="foot"></p>
</div>
<script>
const money=v=>v>=1e6?"$"+(v/1e6).toFixed(1)+"M":v>=1e3?"$"+(v/1e3).toFixed(0)+"k":"$"+Math.round(v);
function ladder(d){
  const rows=[]; const peak=Math.max(...d.bids.map(x=>x[1]),...d.asks.map(x=>x[1]),1);
  const line=(p,v,cls,touch)=>`<tr class="${touch?'touch':''}"><td class="p">${p.toFixed(d.digits)}</td>`+
    `<td class="n">${money(v)}</td><td class="b"><div class="bx" style="width:${Math.round(v/peak*100)}%;background:var(--${cls});opacity:.55"></div></td></tr>`;
  d.asks.slice().reverse().forEach((x,i)=>rows.push(line(x[0],x[1],"sell",i===d.asks.length-1)));
  d.bids.forEach((x,i)=>rows.push(line(x[0],x[1],"buy",i===0)));
  return "<table>"+rows.join("")+"</table>";
}
async function tick(){
  // путь строим от текущего адреса: относительный "state.json" от /<токен>
  // разворачивается в корень и не находится
  const url=location.pathname.replace(/\/+$/,"")+"/state.json";
  let s; try{ s=await (await fetch(url,{cache:"no-store"})).json(); }
  catch(e){ document.getElementById("foot").textContent="нет связи с сервером"; return; }
  const st=s.status;
  document.getElementById("chips").innerHTML=
    `<span class="chip">на диске <b>${st.data_mb} МБ</b></span>`+
    `<span class="chip">свободно <b>${st.free_gb} ГБ</b></span>`+
    `<span class="chip">страница работает <b>${Math.floor(st.watch_uptime/60)} мин</b></span>`;
  document.getElementById("grid").innerHTML=Object.entries(s.symbols).map(([sym,d])=>{
    if(!d.ready) return `<div class="card"><h2>${sym}</h2><p class="meta">собираю книгу…</p></div>`;
    const dir=d.imbalance>0.2?"перевес покупателей":d.imbalance<-0.2?"перевес продавцов":"ровно";
    const delta=d.buy5-d.sell5;
    return `<div class="card"><h2>${sym.replace("_","/")} · ${d.mid.toFixed(d.digits)}</h2>`+
      `<p class="meta">спред ${d.spread_bp.toFixed(3)} б.п. · дисбаланс ${d.imbalance>=0?"+":""}${d.imbalance.toFixed(2)} · ${dir}</p>`+
      ladder(d)+
      `<div class="tape">за 5 с: покупали <span class="up">${money(d.buy5)}</span>, `+
      `продавали <span class="dn">${money(d.sell5)}</span>, дельта `+
      `<b class="${delta>=0?'up':'dn'}">${delta>=0?"+":"−"}${money(Math.abs(delta))}</b> · ${d.count5} сделок`+
      `<div class="last">`+d.last.map(x=>`<span class="${x[1]===1?'up':'dn'}">${x[0].toFixed(d.digits)}</span>`).join(" ")+`</div></div></div>`;
  }).join("");
  document.getElementById("foot").textContent="диктофон: "+(st.recorder||"строка статистики ещё не появилась");
}
tick(); setInterval(tick,1000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    secret = ""

    def _send(self, code, body, ctype):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0].strip("/")
        if path == self.secret:
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == f"{self.secret}/state.json":
            self._send(200, json.dumps(STATE), "application/json")
        elif path == f"{self.secret}/out":
            self._send(200, self._listing(), "text/html; charset=utf-8")
        elif path.startswith(f"{self.secret}/out/"):
            self._send_out(path.rsplit("/", 1)[-1])
        else:
            self._send(404, "not found", "text/plain")

    def _listing(self):
        """Что лежит в out/ — выводы инструментов, сохранённые обёртками."""
        items = []
        for f in sorted(OUT_DIR.glob("*.txt")) if OUT_DIR.exists() else []:
            size = f.stat().st_size
            items.append(f'<li><a href="out/{f.name}">{f.name}</a> '
                         f'— {size/1000:.0f} КБ</li>')
        body = "".join(items) or "<li>пока пусто — запусти любой инструмент</li>"
        return ("<meta charset='utf-8'><title>Выводы инструментов</title>"
                "<body style='font:14px system-ui;padding:20px'>"
                f"<h1>out/</h1><ul>{body}</ul></body>")

    def _send_out(self, name):
        """Только простые имена и только .txt: ни путей, ни выхода из каталога."""
        if "/" in name or "\\" in name or ".." in name or not name.endswith(".txt"):
            self._send(404, "not found", "text/plain")
            return
        target = OUT_DIR / name
        if not target.is_file():
            self._send(404, "not found", "text/plain")
            return
        self._send(200, target.read_text(errors="replace"),
                   "text/plain; charset=utf-8")

    def log_message(self, *args):
        pass          # не засоряем журнал каждым опросом раз в секунду


def serve(secret):
    Handler.secret = secret
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


def main():
    secret = token()
    threading.Thread(target=serve, args=(secret,), daemon=True).start()
    try:
        ip = requests.get("https://ifconfig.me", timeout=5).text.strip()
    except Exception:
        ip = "<IP-сервера>"
    print(f"Страница: http://{ip}:{PORT}/{secret}", flush=True)
    print(f"Выводы инструментов: http://{ip}:{PORT}/{secret}/out", flush=True)
    print("Без токена в адресе страница не открывается. Ссылку не публикуй.",
          flush=True)
    asyncio.run(collector())


if __name__ == "__main__":
    main()
