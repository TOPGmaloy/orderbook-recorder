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

import numpy as np
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
        self.tape = {s: deque(maxlen=4000) for s in SYMBOLS}
        # середина цены раз в секунду — для сигмы и графика
        self.mid_hist = {s: deque(maxlen=1200) for s in SYMBOLS}
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

    def sample_mid(self):
        """Середина цены раз в секунду: из неё считаются сигма и график."""
        now = time.time()
        for s in SYMBOLS:
            b, a = self.books[s].best()
            if not b or not a:
                continue
            hist = self.mid_hist[s]
            if hist and now - hist[-1][0] < 0.9:
                continue
            hist.append((now, (b[0] + a[0]) / 2))

    def volatility(self, symbol):
        """Сигма движения за 60 секунд, в базисных пунктах.

        Считается по фактическим шестидесятисекундным изменениям, а не
        пересчётом из посекундных: цена на коротких интервалах ходит не как
        случайное блуждание, и такой пересчёт занижает результат вдвое.
        """
        hist = list(self.mid_hist[symbol])
        if len(hist) < 90:
            return None
        moves = []
        for i in range(len(hist) - 60):
            t0, m0 = hist[i]
            t1, m1 = hist[i + 60]
            if 55 <= t1 - t0 <= 70 and m0:
                moves.append((m1 / m0 - 1) * 1e4)
        return float(np.std(moves)) if len(moves) >= 30 else None

    def snapshot(self, levels=14):
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

            # сколько денег проторговано на каждой цене за последнюю минуту —
            # это и есть след поглощения: уровень едят, а он стоит
            eaten = {}
            for t, price, vol, side in self.tape[s]:
                if now - t <= 60:
                    eaten[price] = eaten.get(price, 0.0) + vol * unit

            bids = [[p, book.bids[p] * unit, eaten.get(p, 0.0)]
                    for p in sorted(book.bids, reverse=True)[:levels]]
            asks = [[p, book.asks[p] * unit, eaten.get(p, 0.0)]
                    for p in sorted(book.asks)[:levels]]
            recent = [t for t in self.tape[s] if now - t[0] <= 5]
            qb, qa = b[1], a[1]
            sigma = self.volatility(s)
            hist = list(self.mid_hist[s])[-300:]
            out[s] = {
                "ready": True, "mid": mid,
                "spread_bp": (a[0] - b[0]) / mid * 1e4,
                "imbalance": (qb - qa) / (qb + qa) if qb + qa else 0.0,
                "bids": bids, "asks": asks,
                "buy5": sum(t[2] for t in recent if t[3] == 1) * unit,
                "sell5": sum(t[2] for t in recent if t[3] == 2) * unit,
                "count5": len(recent),
                "digits": self.digits.get(s, 2),
                "sigma_bp": sigma,
                # плата за невыгодные исполнения по нашему же замеру:
                # 0.209 от собственной сигмы инструмента (r = 0.966 на шести)
                "cost_bp": sigma * 0.209 if sigma else None,
                "spark": [m for _, m in hist],
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
            live.sample_mid()
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
  :root{
    --surface:#fcfcfb; --plane:#f4f4f1; --ink:#0b0b0b; --ink2:#52514e;
    --muted:#898781; --line:rgba(11,11,11,.12); --grid:#e1e0d9;
    --buy:#2a78d6; --sell:#eb6834; --buy-w:rgba(42,120,214,.30);
    --sell-w:rgba(235,104,52,.30); --touch:rgba(11,11,11,.06);
    color-scheme:light;
  }
  @media (prefers-color-scheme:dark){:root{
    --surface:#151514; --plane:#0b0b0a; --ink:#f2f1ea; --ink2:#c3c2b7;
    --muted:#898781; --line:rgba(255,255,255,.12); --grid:#2c2c2a;
    --buy:#3987e5; --sell:#d95926; --buy-w:rgba(57,135,229,.34);
    --sell-w:rgba(217,89,38,.34); --touch:rgba(255,255,255,.07);
    color-scheme:dark;}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--plane);color:var(--ink);
       font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
  .top{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;
       padding:10px 14px;border-bottom:1px solid var(--line);
       background:var(--surface);position:sticky;top:0;z-index:5}
  .top h1{font-size:14px;font-weight:650;margin:0;letter-spacing:.02em}
  .top .meta{font-size:11.5px;color:var(--muted);
             font-variant-numeric:tabular-nums}
  .wrap{display:grid;gap:10px;padding:10px;
        grid-template-columns:repeat(auto-fit,minmax(292px,1fr))}
  .dom{background:var(--surface);border:1px solid var(--line);border-radius:8px;
       overflow:hidden;display:flex;flex-direction:column}
  .head{display:flex;justify-content:space-between;align-items:baseline;
        padding:7px 10px;border-bottom:1px solid var(--line)}
  .sym{font-weight:650;font-size:12.5px;letter-spacing:.03em}
  .px{font-variant-numeric:tabular-nums;font-weight:650;font-size:14px}
  .sub{display:flex;gap:10px;padding:4px 10px 6px;font-size:11px;
       color:var(--ink2);font-variant-numeric:tabular-nums;flex-wrap:wrap}
  .sub b{color:var(--ink);font-weight:600}
  /* плата за вход — наша величина, её нет ни в одном терминале */
  .fee{margin-left:auto;padding:1px 7px;border-radius:4px;font-size:10.5px;
       border:1px solid var(--line);color:var(--ink2);white-space:nowrap}
  .fee b{font-variant-numeric:tabular-nums}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;
        font-size:11.5px}
  td{padding:0;height:17px;white-space:nowrap;position:relative}
  .eat{width:19%;text-align:center;font-size:9.5px;color:var(--muted)}
  .qty{width:26%;position:relative;overflow:hidden}
  .qty span{position:relative;z-index:2;padding:0 5px;display:block}
  .qty i{position:absolute;top:1px;bottom:1px;display:block;border-radius:2px}
  .l span{text-align:right} .l i{right:0} .l{text-align:right}
  .r span{text-align:left}  .r i{left:0}
  .p{width:30%;text-align:center;color:var(--ink2);
     border-left:1px solid var(--grid);border-right:1px solid var(--grid)}
  tr.at .p{background:var(--touch);color:var(--ink);font-weight:650}
  tr.gap td{height:5px}
  .abs{color:var(--ink);font-weight:700}
  .abs::after{content:"●";font-size:7px;vertical-align:middle;margin-left:2px}
  .tape{display:flex;gap:8px;justify-content:space-between;padding:5px 10px;
        border-top:1px solid var(--line);font-size:11px;color:var(--ink2);
        font-variant-numeric:tabular-nums}
  .up{color:var(--buy)} .dn{color:var(--sell)}
  .spark{display:block;width:100%;height:34px}
  .legend{padding:8px 14px 16px;font-size:11.5px;color:var(--muted);
          line-height:1.6}
  .legend b{color:var(--ink2);font-weight:600}
</style></head><body>
<div class="top">
  <h1>СТАКАН MEXC</h1>
  <span class="meta" id="meta">подключаюсь…</span>
</div>
<div class="wrap" id="wrap"></div>
<p class="legend">
  <b>Лестница.</b> Цена в центре, слева биды, справа аски, ширина полосы —
  объём в деньгах. Крайние колонки — сколько денег проторговано на этой цене
  за последнюю минуту. Точка <span class="abs"></span> у числа значит, что
  съели больше, чем сейчас показано: уровень держат, это след поглощения.
  <br>
  <b>Плата за вход</b> — то, во что обходится пассивное исполнение на этом
  инструменте: 0.209 от его собственной сигмы. Формула не из учебника, а из
  наших замеров на шести инструментах (r = 0.966). Сигнал из стакана стоит
  0.4–1.0 б.п., поэтому всё, что выше, — заведомо в минус.
</p>
<script>
const money=v=>v>=1e6?"$"+(v/1e6).toFixed(1)+"M":v>=1e3?"$"+(v/1e3).toFixed(0)+"k":"$"+Math.round(v);
function spark(points){
  if(!points||points.length<4) return "";
  const lo=Math.min(...points), hi=Math.max(...points), d=(hi-lo)||1;
  const step=100/(points.length-1);
  const path=points.map((v,i)=>`${(i*step).toFixed(2)},${(28-(v-lo)/d*26).toFixed(2)}`).join(" ");
  const up=points[points.length-1]>=points[0];
  return `<svg class="spark" viewBox="0 0 100 30" preserveAspectRatio="none">
    <polyline points="${path}" fill="none" stroke="var(--${up?'buy':'sell'})"
      stroke-width="1.1" vector-effect="non-scaling-stroke"/></svg>`;
}
function ladder(d){
  const all=[...d.bids.map(x=>x[1]),...d.asks.map(x=>x[1])];
  const peak=Math.max(...all,1);
  const dg=d.digits;
  const row=(price,qty,eaten,side,touch)=>{
    const w=Math.max(2,Math.round(qty/peak*100));
    const cls=side==="l"?"buy":"sell";
    const hot=eaten>qty*1.5&&eaten>0?" abs":"";
    const eat=eaten?money(eaten):"";
    const bar=`<i style="width:${w}%;background:var(--${cls}-w)"></i>`;
    const cell=`<td class="qty ${side}">${bar}<span>${money(qty)}</span></td>`;
    const eatCell=`<td class="eat${hot}">${eat}</td>`;
    return side==="l"
      ? `<tr class="${touch?'at':''}">${eatCell}${cell}<td class="p">${price.toFixed(dg)}</td><td class="qty r"></td><td class="eat"></td></tr>`
      : `<tr class="${touch?'at':''}"><td class="eat"></td><td class="qty l"></td><td class="p">${price.toFixed(dg)}</td>${cell}${eatCell}</tr>`;
  };
  let html="";
  d.asks.slice().reverse().forEach((x,i)=>{
    html+=row(x[0],x[1],x[2],"r",i===d.asks.length-1);});
  html+=`<tr class="gap"><td colspan="5"></td></tr>`;
  d.bids.forEach((x,i)=>{html+=row(x[0],x[1],x[2],"l",i===0);});
  return `<table>${html}</table>`;
}
async function tick(){
  const url=location.pathname.replace(/\/+$/,"")+"/state.json";
  let s; try{ s=await (await fetch(url,{cache:"no-store"})).json(); }
  catch(e){ document.getElementById("meta").textContent="нет связи"; return; }
  const st=s.status||{};
  document.getElementById("meta").textContent=
    `${Object.keys(s.symbols||{}).length} инструментов · на диске ${st.data_mb} МБ · `+
    `свободно ${st.free_gb} ГБ · ${(st.recorder||"диктофон молчит").slice(0,90)}`;
  document.getElementById("wrap").innerHTML=Object.entries(s.symbols).map(([sym,d])=>{
    if(!d.ready) return `<div class="dom"><div class="head"><span class="sym">${sym}</span></div>
      <div class="sub">собираю книгу…</div></div>`;
    const delta=d.buy5-d.sell5;
    const imb=d.imbalance;
    const fee=d.cost_bp!=null
      ? `<span class="fee" title="0.209 от сигмы инструмента — наш замер">плата за вход <b>${d.cost_bp.toFixed(2)}</b> б.п.</span>`
      : `<span class="fee">плата за вход — считаю…</span>`;
    return `<div class="dom">
      <div class="head"><span class="sym">${sym.replace("_","/")}</span>
        <span class="px">${d.mid.toFixed(d.digits)}</span></div>
      <div class="sub">
        <span>спред <b>${d.spread_bp.toFixed(3)}</b></span>
        <span>перекос <b class="${imb>0?'up':'dn'}">${imb>=0?"+":""}${imb.toFixed(2)}</b></span>
        ${d.sigma_bp!=null?`<span>сигма <b>${d.sigma_bp.toFixed(1)}</b></span>`:""}
        ${fee}
      </div>
      ${ladder(d)}
      <div class="tape">
        <span>за 5 с <span class="up">${money(d.buy5)}</span> / <span class="dn">${money(d.sell5)}</span></span>
        <span>дельта <b class="${delta>=0?'up':'dn'}">${delta>=0?"+":"−"}${money(Math.abs(delta))}</b></span>
      </div>
      ${spark(d.spark)}
    </div>`;
  }).join("");
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
