#!/usr/bin/env python3
"""Просмотрщик записи: тепловая карта стакана + лента + лестница цен.

    python tools/viewer.py --symbol BTC_USDT --minutes 8 --out viewer.html

Делает из записанного один самодостаточный HTML-файл: слева тепловая карта
(время по горизонтали, цена по вертикали, яркость — сколько стоит объёма),
поверх неё точки сделок, справа — лестница стакана на тот момент, куда
навёл мышь. Ничего никуда не отправляет, интернет для просмотра не нужен.

Зачем: глазами видно то, чего не видно в отчёте — поглощения, сбежавшие
плиты, следы айсбергов.
"""

import argparse
import base64
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa
import pyarrow.parquet as pq

from config import DATA_DIR
from recorder.book import OrderBook


def read_closed(files):
    """Читает parquet, пропуская тот файл, в который сейчас идёт запись.

    У parquet footer дописывается при закрытии файла, поэтому текущий файл
    прочитать нельзя — он станет доступен после ротации (каждые
    ROTATE_MINUTES) или после остановки службы. Это не ошибка, а нормальная
    работа: молча пропускаем его.
    """
    tables, skipped = [], []
    for f in files:
        try:
            tables.append(pq.read_table(f))
        except Exception:
            skipped.append(f.name)
    if skipped:
        print(f"пропущен файл в работе: {', '.join(skipped)}", file=sys.stderr)
    if not tables:
        sys.exit("Ни одного закрытого файла — подожди ротации или останови службу.")
    return pa.concat_tables(tables)


def contract_size(symbol):
    """Сколько монет в одном контракте MEXC.

    Объёмы в стакане и ленте приходят В КОНТРАКТАХ, а не в монетах и не в
    долларах: у BTC_USDT контракт равен 0.0001 BTC, у ETH_USDT — 0.01 ETH.
    Без этой поправки «плита 279k» читается как чушь, а на деле это 27.9 BTC.
    """
    fallback = {"BTC_USDT": 0.0001, "ETH_USDT": 0.01, "SOL_USDT": 0.1}
    try:
        import requests
        body = requests.get("https://contract.mexc.com/api/v1/contract/detail",
                            timeout=15).json()
        for row in body.get("data") or []:
            if row.get("symbol") == symbol:
                return float(row["contractSize"])
    except Exception as exc:
        print(f"размер контракта не получен ({exc}) — беру из таблицы",
              file=sys.stderr)
    return fallback.get(symbol, 1.0)


def load_rows(symbol):
    files = sorted(DATA_DIR.rglob("events_*.parquet"))
    if not files:
        sys.exit(f"В {DATA_DIR} пусто — сначала запиши данные.")
    table = read_closed(files)
    rows = [r for r in table.to_pylist() if r["symbol"] in (symbol, "")]
    rows.sort(key=lambda r: r["ts_local_us"])
    return [r for r in rows if r["symbol"] == symbol]


def build(symbol, minutes, step_ms, bins, depth_rows):
    rows = load_rows(symbol)
    if not rows:
        sys.exit(f"Нет записей по {symbol}.")

    end_us = rows[-1]["ts_local_us"]
    start_us = end_us - minutes * 60 * 1_000_000
    rows = [r for r in rows if r["ts_local_us"] >= start_us]
    if len(rows) < 10:
        sys.exit("Слишком мало данных в окне — увеличь --minutes.")
    start_us = rows[0]["ts_local_us"]

    # --- первый проход: собираем книгу и снимаем её состояние по сетке ------
    book = OrderBook(symbol)
    frames = []          # (ts_us, {price: size}, {price: orders}) для бида и аска
    trades = []          # (ts_us, price, volume, side)  side: 0 покупка, 1 продажа
    next_ts = None          # ставится, когда книга впервые собралась
    step_us = step_ms * 1000

    def snap():
        bid = {p: v for p, v in book.bids.items()}
        ask = {p: v for p, v in book.asks.items()}
        frames.append((next_ts, bid, ask))

    for r in rows:
        while next_ts is not None and r["ts_local_us"] >= next_ts + step_us:
            snap()
            next_ts += step_us
        payload = json.loads(r["payload"])
        if r["channel"] == "snapshot":
            book.apply_snapshot(payload)
            if next_ts is None and book.ready:
                # сетка кадров начинается там, где книга стала настоящей,
                # иначе первые кадры были бы копиями одного состояния
                next_ts = r["ts_local_us"]
                start_us = next_ts
        elif r["channel"] == "depth":
            book.apply_delta(payload)
        elif r["channel"] == "deal":
            for deal in (payload if isinstance(payload, list) else [payload]):
                try:
                    trades.append((r["ts_local_us"], float(deal["p"]),
                                   float(deal["v"]), 0 if int(deal["T"]) == 1 else 1))
                except (KeyError, TypeError, ValueError):
                    continue

    if len(frames) < 5:
        sys.exit("Книга не собралась на этом окне — проверь отчётом report.py.")

    # --- ценовая сетка ------------------------------------------------------
    mids = []
    for _, bid, ask in frames:
        if bid and ask:
            mids.append((max(bid) + min(ask)) / 2)
    if not mids:
        sys.exit("Нет ни одного кадра с двусторонней книгой.")
    lo, hi = min(mids), max(mids)
    center = (lo + hi) / 2
    # окно по цене: либо размах за период с запасом, либо минимум 0.08% —
    # чтобы на спокойном рынке было видно структуру у лучшей цены
    half = max((hi - lo) * 0.9, center * 0.0008)
    p_lo, p_hi = center - half, center + half
    step_price = (p_hi - p_lo) / bins

    def to_bin(price):
        idx = int((price - p_lo) / step_price)
        return idx if 0 <= idx < bins else None

    n = len(frames)
    bid_size = bytearray(n * bins)
    ask_size = bytearray(n * bins)
    raw_bid = [[0.0] * bins for _ in range(n)]
    raw_ask = [[0.0] * bins for _ in range(n)]
    best = []

    for fi, (ts, bid, ask) in enumerate(frames):
        for price, volume in bid.items():
            b = to_bin(price)
            if b is not None:
                raw_bid[fi][b] += volume
        for price, volume in ask.items():
            b = to_bin(price)
            if b is not None:
                raw_ask[fi][b] += volume
        bb = to_bin(max(bid)) if bid else None
        ba = to_bin(min(ask)) if ask else None
        best.append([bb if bb is not None else -1, ba if ba is not None else -1])

    # --- сырые объёмы: uint32, из них же считается яркость в браузере -------
    import struct
    def pack(matrix):
        flat = bytearray()
        for row in matrix:
            for v in row:
                flat += struct.pack("<I", min(int(v), 0xFFFFFFFF))
        return base64.b64encode(bytes(flat)).decode()

    pool = sorted(v for row in raw_bid + raw_ask for v in row if v > 0)
    if not pool:
        sys.exit("В окне нет объёма — данные пустые.")
    ref = pool[min(len(pool) - 1, int(len(pool) * 0.99))]

    packed = []
    for ts, price, volume, side in trades:
        fi = int((ts - start_us) / step_us)
        b = to_bin(price)
        if 0 <= fi < n and b is not None:
            packed.append([fi, b, round(volume), side])
    vols = sorted(t[2] for t in packed)
    vol_ref = vols[int(len(vols) * 0.95)] if vols else 1

    return {
        "symbol": symbol,
        "contract": contract_size(symbol),
        "start_ms": start_us // 1000,
        "step_ms": step_ms,
        "frames": n,
        "bins": bins,
        "p_lo": p_lo,
        "p_step": step_price,
        "depth_rows": depth_rows,
        "size_ref": ref,
        "vol_ref": max(vol_ref, 1),
        "bid": pack(raw_bid),
        "ask": pack(raw_ask),
        "best": best,
        "trades": packed,
        "trade_count": len(packed),
    }


TEMPLATE = r"""<title>__TITLE__</title>
<style>
  .obr {
    --surface: #fcfcfb; --plane: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --line: rgba(11,11,11,0.10);
    --buy: #2a78d6; --sell: #eb6834;
    --heat-lo: #eeece6; --heat-hi: #0b0b0b;
    color-scheme: light;
    background: var(--plane); color: var(--ink);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 20px; max-width: 1240px; margin: 0 auto;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) .obr {
      --surface: #1a1a19; --plane: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
      --muted: #898781; --grid: #2c2c2a; --line: rgba(255,255,255,0.10);
      --buy: #3987e5; --sell: #d95926;
      --heat-lo: #232322; --heat-hi: #f2f1ea;
      color-scheme: dark;
    }
  }
  :root[data-theme="dark"] .obr {
    --surface: #1a1a19; --plane: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --line: rgba(255,255,255,0.10);
    --buy: #3987e5; --sell: #d95926;
    --heat-lo: #232322; --heat-hi: #f2f1ea;
    color-scheme: dark;
  }
  .obr h1 { font-size: 19px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
  .obr .sub { color: var(--ink-2); margin: 0 0 16px; font-size: 13px; }
  .obr .chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 16px; }
  .obr .chip {
    background: var(--surface); border: 1px solid var(--line); border-radius: 6px;
    padding: 4px 9px; font-size: 12px; color: var(--ink-2);
    font-variant-numeric: tabular-nums;
  }
  .obr .chip b { color: var(--ink); font-weight: 600; }
  .obr .stage { display: flex; gap: 14px; align-items: stretch; flex-wrap: wrap; }
  .obr .plot {
    flex: 1 1 520px; min-width: 0; background: var(--surface);
    border: 1px solid var(--line); border-radius: 10px; padding: 8px;
  }
  .obr canvas { display: block; width: 100%; touch-action: none; }
  .obr .side { flex: 0 0 268px; min-width: 0; }
  @media (max-width: 760px) { .obr .side { flex: 1 1 100%; } }
  .obr .card {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 10px; padding: 10px 12px;
  }
  .obr .card h2 {
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--muted); margin: 0 0 8px;
  }
  .obr table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  .obr td { padding: 1px 4px; font-size: 12px; white-space: nowrap; }
  .obr td.p { color: var(--ink-2); width: 34%; }
  .obr td.n { text-align: right; width: 22%; color: var(--ink); }
  .obr td.bar { width: 44%; }
  .obr .bx { height: 9px; border-radius: 2px; }
  .obr tr.touch td { border-top: 1px solid var(--muted); }
  .obr tr.touch td.p { color: var(--ink); font-weight: 600; }
  .obr .legend {
    display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
    margin: 14px 0 0; font-size: 12px; color: var(--ink-2);
  }
  .obr .key { display: inline-flex; align-items: center; gap: 6px; }
  .obr .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .obr .ramp {
    width: 84px; height: 9px; border-radius: 2px;
    background: linear-gradient(90deg, var(--heat-lo), var(--heat-hi));
  }
  .obr .note {
    margin: 16px 0 0; font-size: 12.5px; color: var(--ink-2);
    border-top: 1px solid var(--line); padding-top: 12px;
  }
  .obr .note b { color: var(--ink); font-weight: 600; }
</style>

<div class="obr">
  <h1>__TITLE__</h1>
  <p class="sub">__SUB__</p>
  <div class="chips">__CHIPS__</div>

  <div class="stage">
    <div class="plot"><canvas id="cv" height="600"></canvas></div>
    <div class="side">
      <div class="card">
        <h2 id="lad-t">Стакан на конец окна</h2>
        <p style="margin:-4px 0 8px;font-size:11px;color:var(--muted)">
          объём в деньгах по номиналу</p>
        <table id="lad"></table>
      </div>
    </div>
  </div>

  <div class="legend">
    <span class="key"><span class="ramp"></span> объём в стакане: меньше → больше</span>
    <span class="key"><span class="dot" style="background:var(--buy)"></span> сделка: покупатель забрал по аску</span>
    <span class="key"><span class="dot" style="background:var(--sell)"></span> сделка: продавец отдал по биду</span>
    <span class="key">— тонкие линии: лучший бид и лучший аск</span>
  </div>

  <p class="note">
    <b>Единицы.</b> Биржа отдаёт объёмы в контрактах, а не в монетах:
    у BTC_USDT один контракт равен 0.0001 BTC. В лестнице справа пересчитано
    в деньги — «$850k» значит, что на этой цене стоит заявок на 850 тысяч
    долларов. Наведи на строку, чтобы увидеть контракты и монеты.
    <br><br>
    <b>Как читать.</b> Тёмные горизонтальные полосы — плиты: объём, который
    стоит на одной цене. Плита, в которую бьют сделки и которая не тает, —
    это <b>поглощение</b>: кто-то крупный скупает всё, что в него льют.
    Плита, исчезнувшая до подхода цены, — <b>сбежала</b>, опоры там нет.
    Наведи курсор — справа появится стакан на тот момент.
  </p>
</div>

<script>
(function () {
  const D = __DATA__;
  const cv = document.getElementById("cv");
  const ctx = cv.getContext("2d");
  const ladder = document.getElementById("lad");
  const ladderTitle = document.getElementById("lad-t");

  const un64 = (s) => {
    const bin = atob(s), out = new Uint32Array(bin.length / 4);
    const dv = new DataView(new ArrayBuffer(bin.length));
    for (let i = 0; i < bin.length; i++) dv.setUint8(i, bin.charCodeAt(i));
    for (let i = 0; i < out.length; i++) out[i] = dv.getUint32(i * 4, true);
    return out;
  };
  const BID = un64(D.bid), ASK = un64(D.ask);
  const F = D.frames, B = D.bins;
  // Корневая шкала, не логарифмическая: у объёмов в стакане разброс всего
  // порядка 10x, и логарифм сжимает его в сплошное тёмное пятно.
  const shade = (v) => Math.min(1, Math.sqrt(v / D.size_ref));

  const css = (n) => getComputedStyle(cv).getPropertyValue(n).trim();
  const hex = (h) => {
    h = h.replace("#", "");
    if (h.length === 3) h = h.split("").map(c => c + c).join("");
    return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
  };
  const fmt = (v) => v >= 1e6 ? (v/1e6).toFixed(1)+"M" : v >= 1e3 ? (v/1e3).toFixed(0)+"k" : String(Math.round(v));
  // Объёмы приходят в контрактах: у BTC_USDT контракт = 0.0001 BTC. Людям
  // нужны деньги, а не контракты, поэтому в лестнице показываем номинал.
  const usd = (contracts, px) => "$" + fmt(contracts * D.contract * px);
  const price = (b) => D.p_lo + (b + 0.5) * D.p_step;
  const clock = (f) => new Date(D.start_ms + f * D.step_ms)
        .toISOString().slice(11, 19);

  const M = { l: 58, r: 10, t: 10, b: 26 };
  const TAPE = 84, GAPY = 12;
  let plot = {}, hover = null, heatCanvas = null;

  function buildHeat() {
    const lo = hex(css("--heat-lo")), hi = hex(css("--heat-hi"));
    const img = ctx.createImageData(F, B);
    for (let f = 0; f < F; f++) {
      for (let b = 0; b < B; b++) {
        const v = BID[f*B+b] + ASK[f*B+b];
        const y = B - 1 - b, i = (y * F + f) * 4;
        if (v > 0) {
          const t = shade(v);
          img.data[i]   = lo[0] + (hi[0]-lo[0]) * t;
          img.data[i+1] = lo[1] + (hi[1]-lo[1]) * t;
          img.data[i+2] = lo[2] + (hi[2]-lo[2]) * t;
          img.data[i+3] = 255;
        } else { img.data[i+3] = 0; }
      }
    }
    heatCanvas = document.createElement("canvas");
    heatCanvas.width = F; heatCanvas.height = B;
    heatCanvas.getContext("2d").putImageData(img, 0, 0);
  }

  function layout() {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = 600;
    cv.width = w * dpr; cv.height = h * dpr;
    cv.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    plot = { x: M.l, y: M.t, w: w - M.l - M.r,
             h: h - M.t - M.b - TAPE - GAPY, tapeY: h - M.b - TAPE };
  }

  const fx = (f) => plot.x + (f + 0.5) / F * plot.w;
  const fy = (b) => plot.y + (1 - (b + 0.5) / B) * plot.h;

  function draw() {
    const w = cv.clientWidth, h = 600;
    ctx.clearRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(heatCanvas, plot.x, plot.y, plot.w, plot.h);

    // лучший бид и аск — тонкие линии, они же показывают спред
    for (const pass of [0, 1]) {
      ctx.lineWidth = pass === 0 ? 3.5 : 1.5;
      ctx.strokeStyle = pass === 0 ? css("--surface") : css("--ink-2");
      ctx.globalAlpha = pass === 0 ? 0.8 : 1;
      for (const side of [0, 1]) {
        ctx.beginPath();
        let started = false;
        for (let f = 0; f < F; f++) {
          const b = D.best[f][side];
          if (b < 0) { started = false; continue; }
          const X = fx(f), Y = fy(b);
          started ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
          started = true;
        }
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    // сделки: площадь кружка пропорциональна объёму, обводка цветом фона
    ctx.lineWidth = 2; ctx.strokeStyle = css("--surface");
    for (const [f, b, v, side] of D.trades) {
      const r = Math.min(9, 1.8 + 5.5 * Math.sqrt(v / D.vol_ref));
      ctx.beginPath(); ctx.arc(fx(f), fy(b), r, 0, 6.2832);
      ctx.fillStyle = side === 0 ? css("--buy") : css("--sell");
      ctx.globalAlpha = 0.85; ctx.fill();
      ctx.globalAlpha = 1; if (r > 3) ctx.stroke();
    }

    drawTape();
    drawAxes();
    if (hover !== null) drawCross();
  }

  function drawTape() {
    const mid = plot.tapeY + TAPE / 2;
    const buy = new Float64Array(F), sell = new Float64Array(F);
    for (const [f, , v, side] of D.trades) (side === 0 ? buy : sell)[f] += v;
    let peak = 1;
    for (let f = 0; f < F; f++) peak = Math.max(peak, buy[f], sell[f]);
    const bw = Math.max(1, plot.w / F - 2);
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(plot.x, mid); ctx.lineTo(plot.x + plot.w, mid); ctx.stroke();
    for (let f = 0; f < F; f++) {
      const X = fx(f) - bw / 2;
      const H = TAPE / 2 - 8;
      if (buy[f]) {
        const h = Math.max(1.5, Math.sqrt(buy[f] / peak) * H);
        ctx.fillStyle = css("--buy"); ctx.fillRect(X, mid - h, bw, h);
      }
      if (sell[f]) {
        const h = Math.max(1.5, Math.sqrt(sell[f] / peak) * H);
        ctx.fillStyle = css("--sell"); ctx.fillRect(X, mid + 2, bw, h);
      }
    }
    ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "left"; ctx.fillText("лента: объём сделок", plot.x + 4, plot.tapeY + 11);
  }

  function drawAxes() {
    ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const steps = 6;
    for (let i = 0; i <= steps; i++) {
      const b = Math.round(i / steps * (B - 1));
      ctx.fillText(price(b).toFixed(D.p_step < 0.1 ? 2 : 1), plot.x - 7, fy(b));
      ctx.strokeStyle = css("--grid"); ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.moveTo(plot.x - 3, fy(b)); ctx.lineTo(plot.x, fy(b)); ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let i = 0; i <= 5; i++) {
      const f = Math.round(i / 5 * (F - 1));
      ctx.fillText(clock(f), fx(f), 600 - M.b + 6);
    }
  }

  function drawCross() {
    const X = fx(hover);
    ctx.save();
    ctx.strokeStyle = css("--ink"); ctx.globalAlpha = 0.45;
    ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X, plot.y); ctx.lineTo(X, plot.y + plot.h); ctx.stroke();
    ctx.restore();
  }

  function renderLadder(f) {
    const bb = D.best[f][0], ba = D.best[f][1];
    const anchor = bb >= 0 && ba >= 0 ? (bb + ba) / 2 : Math.floor(B / 2);
    const half = D.depth_rows;
    const from = Math.max(0, Math.round(anchor - half));
    const to = Math.min(B - 1, Math.round(anchor + half));
    let peak = 1;
    for (let b = from; b <= to; b++) peak = Math.max(peak, BID[f*B+b], ASK[f*B+b]);
    const rows = [];
    for (let b = to; b >= from; b--) {
      const bv = BID[f*B+b], av = ASK[f*B+b], v = bv + av;
      const isAsk = av >= bv;
      const cls = (b === bb || b === ba) ? ' class="touch"' : "";
      const pct = Math.round(v / peak * 100);
      const px = price(b);
      const tip = v ? ` title="${fmt(v)} контрактов = ${(v*D.contract).toFixed(4)} ${D.coin}"` : "";
      rows.push(`<tr${cls}${tip}><td class="p">${px.toFixed(D.p_step < 0.1 ? 2 : 1)}</td>` +
        `<td class="n">${v ? usd(v, px) : ""}</td><td class="bar">` +
        (v ? `<div class="bx" style="width:${pct}%;background:${isAsk ? "var(--sell)" : "var(--buy)"};opacity:.55"></div>` : "") +
        `</td></tr>`);
    }
    ladder.innerHTML = rows.join("");
    ladderTitle.textContent = "Стакан в " + clock(f);
  }

  cv.addEventListener("pointermove", (e) => {
    const r = cv.getBoundingClientRect();
    const f = Math.round(((e.clientX - r.left) - plot.x) / plot.w * F - 0.5);
    if (f >= 0 && f < F) { hover = f; renderLadder(f); draw(); }
  });
  cv.addEventListener("pointerleave", () => { hover = null; renderLadder(F - 1); draw(); });

  function boot() { layout(); buildHeat(); renderLadder(F - 1); draw(); }
  boot();
  new ResizeObserver(() => { layout(); draw(); }).observe(cv);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", boot);
})();
</script>
"""


def render(data, fragment=False):
    started = datetime.fromtimestamp(data["start_ms"] / 1000, timezone.utc)
    minutes = data["frames"] * data["step_ms"] / 60000
    title = f"Стакан {data['symbol'].replace('_', '/')}"
    sub = (f"Запись MEXC futures, {started:%Y-%m-%d %H:%M} UTC, "
           f"{minutes:.0f} мин, кадр {data['step_ms']} мс")
    coin = data["symbol"].split("_")[0]
    data["coin"] = coin
    chips = "".join(
        f'<span class="chip">{k} <b>{v}</b></span>' for k, v in [
            ("кадров", data["frames"]),
            ("сделок", data["trade_count"]),
            ("контракт", f"{data['contract']:g} {coin}"),
            ("шаг цены", f"{data['p_step']:.2f}"),
            ("диапазон", f"{data['p_lo']:.1f} – {data['p_lo'] + data['p_step']*data['bins']:.1f}"),
        ])
    html = (TEMPLATE.replace("__TITLE__", title).replace("__SUB__", sub)
            .replace("__CHIPS__", chips)
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    if fragment:
        return html
    return ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<style>*{box-sizing:border-box}body{margin:0}</style>"
            "</head><body>" + html + "</body></html>")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTC_USDT")
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--step-ms", type=int, default=400)
    ap.add_argument("--bins", type=int, default=160)
    ap.add_argument("--depth-rows", type=int, default=13)
    ap.add_argument("--out", default="viewer.html")
    ap.add_argument("--fragment", action="store_true",
                    help="без обвязки <html>: для публикации артефактом")
    a = ap.parse_args()

    data = build(a.symbol, a.minutes, a.step_ms, a.bins, a.depth_rows)
    Path(a.out).write_text(render(data, a.fragment), encoding="utf-8")
    size = Path(a.out).stat().st_size / 1e6
    print(f"{a.out}  —  {data['frames']} кадров, {data['trade_count']} сделок, "
          f"{size:.1f} МБ")


if __name__ == "__main__":
    main()
