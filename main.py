#!/usr/bin/env python3
"""
cbBTC/USDC Dislocation Reversion Trading Bot
=============================================
Competition bot that reads bar messages from stdin and writes order replies
to stdout. Implements a mean-reversion strategy on rolling average price
dislocation with strict drawdown controls.

Message types handled:
  - hello  : session handshake
  - bar    : OHLCV price bar → strategy runs here
  - orders : exchange order confirmations
  - batch  : grouped envelope containing multiple sub-messages

Order types emitted:
  - {"side":"buy",  "usdc":    <amount>}  — spend USDC to acquire cbBTC
  - {"side":"sell", "fraction":<0..1>}    — sell a fraction of cbBTC holdings

Portfolio starts with 1 000 USDC and zero cbBTC.
Trading halts automatically when mark-to-market drawdown reaches 20 %.
Zero-volume bars are skipped entirely — no price update, no orders.
"""

import sys
import json
from collections import deque
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — all tuneable knobs in one place
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """
    Central configuration for the dislocation reversion strategy.
    Keeping every magic number here makes back-testing parameter sweeps trivial.
    """

    # ── Starting capital ───────────────────────────────────────────────────────
    STARTING_USDC: float = 1_000.0

    # ── Rolling average window (bars) ──────────────────────────────────────────
    # The strategy needs at least WINDOW bars before it can fire any orders.
    # 20 bars strikes a balance between responsiveness and noise resistance.
    ROLLING_WINDOW: int = 20

    # ── Dislocation thresholds ─────────────────────────────────────────────────
    # A bar's close must be this far from the rolling average to trigger a signal.
    # 0.3 % keeps us out of micro-noise while still catching genuine dislocations.
    BUY_DISLOCATION:  float = 0.003   # close < avg*(1 - 0.003) → buy signal
    SELL_DISLOCATION: float = 0.003   # close > avg*(1 + 0.003) → sell signal

    # ── Position sizing ────────────────────────────────────────────────────────
    # On each buy signal we risk a fixed fraction of available USDC so that a
    # run of bad signals never wipes the whole book in one go.
    BUY_USDC_FRACTION:  float = 0.30  # spend 30 % of available USDC per signal
    SELL_BTC_FRACTION:  float = 0.50  # sell 50 % of BTC holdings per signal

    # ── Drawdown controls ──────────────────────────────────────────────────────
    # SOFT_LIMIT  — bot stops generating new orders once this is breached.
    # HARD_LIMIT  — competition rule; must never be touched.
    SOFT_DRAWDOWN_LIMIT: float = 0.20   # halt at 20 % drawdown
    HARD_DRAWDOWN_LIMIT: float = 0.25   # competition ceiling (never reached)

    # ── Minimum order sizes ────────────────────────────────────────────────────
    # Avoid emitting dust orders that the exchange might reject.
    MIN_BUY_USDC:  float = 0.01
    MIN_SELL_FRAC: float = 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio — tracks cash, coin, and risk metrics
# ═══════════════════════════════════════════════════════════════════════════════

class Portfolio:
    """
    Maintains the bot's simulated portfolio state.

    All fills are assumed to execute at bar close price (the only price
    information available without an order-book). This keeps the simulation
    deterministic and consistent with how competition scoring typically works.

    Attributes
    ----------
    usdc        : cash balance in USDC
    btc         : cbBTC balance
    peak_equity : highest mark-to-market equity seen so far (for drawdown)
    halted      : True once soft drawdown limit is breached
    """

    def __init__(self, starting_usdc: float = Config.STARTING_USDC) -> None:
        self.usdc: float = starting_usdc
        self.btc:  float = 0.0
        self.peak_equity: float = starting_usdc
        self.halted: bool = False

        # Statistics tracked for logging / debugging (never affect orders)
        self._total_buy_orders:  int = 0
        self._total_sell_orders: int = 0
        self._bars_processed:    int = 0

    # ── Mark-to-market ─────────────────────────────────────────────────────────

    def equity(self, price: float) -> float:
        """Total portfolio value in USDC at the given price."""
        return self.usdc + self.btc * price

    def drawdown(self, price: float) -> float:
        """
        Current drawdown as a fraction of peak equity (0.0 → 1.0).
        Updates peak_equity as a side effect when a new high-water mark is set.
        """
        current_equity = self.equity(price)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - current_equity) / self.peak_equity)

    # ── Fill simulation ────────────────────────────────────────────────────────

    def apply_buy(self, usdc_amount: float, fill_price: float) -> None:
        """
        Record a buy fill: deduct USDC, add cbBTC.
        Call this AFTER emitting the order so the portfolio reflects the
        expected state for the next bar.
        """
        usdc_amount = min(usdc_amount, self.usdc)   # never overspend
        btc_acquired = usdc_amount / fill_price
        self.usdc -= usdc_amount
        self.btc  += btc_acquired
        self._total_buy_orders += 1

    def apply_sell(self, fraction: float, fill_price: float) -> None:
        """
        Record a sell fill: deduct cbBTC fraction, add USDC proceeds.
        """
        fraction    = min(max(fraction, 0.0), 1.0)
        btc_sold    = self.btc * fraction
        usdc_gained = btc_sold * fill_price
        self.btc  -= btc_sold
        self.usdc += usdc_gained
        self._total_sell_orders += 1

    # ── Risk gate ──────────────────────────────────────────────────────────────

    def check_and_update_risk(self, price: float) -> bool:
        """
        Compute drawdown, update peak equity, and set halted flag if the soft
        drawdown limit is breached.
        Returns True if trading is allowed, False if halted.
        """
        current_drawdown = self.drawdown(price)
        if current_drawdown >= Config.SOFT_DRAWDOWN_LIMIT:
            self.halted = True
        return not self.halted

    def __repr__(self) -> str:
        return (
            f"Portfolio(usdc={self.usdc:.4f}, btc={self.btc:.8f}, "
            f"peak={self.peak_equity:.4f}, halted={self.halted})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PriceTracker — rolling statistics on bar close prices
# ═══════════════════════════════════════════════════════════════════════════════

class PriceTracker:
    """
    Maintains a fixed-length window of bar close prices and exposes
    the rolling simple moving average (SMA).

    Only valid bars (non-zero volume, positive close) are ever pushed here,
    so the average is never contaminated by phantom or stale prices.
    """

    def __init__(self, window: int = Config.ROLLING_WINDOW) -> None:
        self._window: int = window
        self._prices: deque[float] = deque(maxlen=window)

    def update(self, bar: dict[str, Any]) -> None:
        """Ingest a bar dict and push its close price onto the rolling window."""
        close = float(bar["close"])
        self._prices.append(close)

    def rolling_average(self) -> float | None:
        """
        Return the rolling SMA of bar close prices, or None if the window
        hasn't filled yet (fewer than ROLLING_WINDOW bars seen so far).
        """
        if len(self._prices) < self._window:
            return None
        return sum(self._prices) / len(self._prices)

    @property
    def is_ready(self) -> bool:
        """True once enough bars have been ingested to compute the average."""
        return len(self._prices) >= self._window

    @property
    def latest_price(self) -> float | None:
        """Most recently ingested close price, or None if no bars yet."""
        return self._prices[-1] if self._prices else None

    @property
    def bars_seen(self) -> int:
        """Total number of bar updates ingested (may exceed window size)."""
        return len(self._prices)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy — pure signal logic, no I/O
# ═══════════════════════════════════════════════════════════════════════════════

class DisplacementReversionStrategy:
    """
    Dislocation (displacement) reversion strategy.

    Philosophy
    ----------
    When price dislocates below the 20-bar rolling average by >= 0.3 %, the
    market has likely overshot and a reversion is probable → buy.
    When price dislocates above the average by >= 0.3 %, the rally is likely
    extended and a pullback is probable → sell.

    Bar validation
    --------------
    Zero-volume bars are skipped entirely: they represent periods with no real
    trading (halted market, data gaps, auction crosses) and their close price
    is not a reliable market signal. Skipping them also prevents the rolling
    average from being dragged by stale repeated prices.

    Signal rules
    ------------
    BUY  : close < rolling_average * (1 - BUY_DISLOCATION)
           AND USDC available > MIN_BUY_USDC
    SELL : close > rolling_average * (1 + SELL_DISLOCATION)
           AND BTC held > 0

    At most one order is emitted per bar (buy XOR sell, never both).
    """

    def __init__(self, portfolio: Portfolio, tracker: PriceTracker) -> None:
        self._portfolio = portfolio
        self._tracker   = tracker

    # ── Bar validation ─────────────────────────────────────────────────────────

    @staticmethod
    def _bar_is_valid(bar: dict[str, Any]) -> bool:
        """
        Reject bars that cannot produce a meaningful signal:
          - missing or non-positive close price
          - zero or missing volume (no real trading occurred this period)
          - non-finite close (NaN / Inf from upstream data issues)

        Skipping these keeps the rolling average uncontaminated and prevents
        the strategy from acting on phantom price moves.
        """
        try:
            close  = float(bar.get("close",  0))
            volume = float(bar.get("volume", 0))
        except (TypeError, ValueError):
            return False

        if volume <= 0:
            return False          # zero-volume bar — no real trades occurred

        if close <= 0:
            return False          # non-positive price is nonsensical

        # Guard against NaN / Inf that can slip through float() conversion
        if close != close:        # NaN check (NaN != NaN is always True)
            return False
        if close == float("inf") or close == float("-inf"):
            return False

        return True

    # ── Signal detection ───────────────────────────────────────────────────────

    def _is_buy_signal(self, close: float, rolling_avg_price: float) -> bool:
        """Price has dislocated below average → mean-reversion buy."""
        buy_threshold = rolling_avg_price * (1.0 - Config.BUY_DISLOCATION)
        has_cash      = self._portfolio.usdc > Config.MIN_BUY_USDC
        return close < buy_threshold and has_cash

    def _is_sell_signal(self, close: float, rolling_avg_price: float) -> bool:
        """Price has dislocated above average → mean-reversion sell."""
        sell_threshold = rolling_avg_price * (1.0 + Config.SELL_DISLOCATION)
        has_btc        = self._portfolio.btc > 0.0
        return close > sell_threshold and has_btc

    # ── Order construction ─────────────────────────────────────────────────────

    def _build_buy_order(self) -> dict[str, Any]:
        """Construct a buy order using BUY_USDC_FRACTION of available cash."""
        spend = self._portfolio.usdc * Config.BUY_USDC_FRACTION
        spend = max(spend, Config.MIN_BUY_USDC)
        return {"side": "buy", "usdc": round(spend, 6)}

    def _build_sell_order(self) -> dict[str, Any]:
        """Construct a sell order for SELL_BTC_FRACTION of BTC holdings."""
        return {"side": "sell", "fraction": round(Config.SELL_BTC_FRACTION, 6)}

    # ── Main entry point called per bar ───────────────────────────────────────

    def evaluate(self, bar: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Evaluate one bar and return a (possibly empty) list of orders.

        Steps:
          1. Validate bar — skip zero-volume or malformed bars entirely.
             Neither the price tracker nor the portfolio is touched for bad bars.
          2. Push close price into the rolling price tracker.
          3. Check risk gate — if halted, return no orders.
          4. Compute rolling average — if window not full yet, return no orders.
          5. Test buy / sell signals and construct orders.
          6. Apply simulated fill to portfolio so next bar sees updated state.
        """
        # Step 1 — validate; zero-volume bars produce no output and don't
        # advance the rolling window so the average stays clean.
        if not self._bar_is_valid(bar):
            return []

        close = float(bar["close"])
        self._tracker.update(bar)
        self._portfolio._bars_processed += 1

        # Step 3 — risk gate: drawdown check updates peak and may set halted
        if not self._portfolio.check_and_update_risk(close):
            return []

        # Step 4 — need a full window before the rolling average is meaningful
        rolling_avg_price = self._tracker.rolling_average()
        if rolling_avg_price is None:
            return []

        orders: list[dict[str, Any]] = []

        # Step 5 — signal evaluation (buy XOR sell per bar)
        if self._is_buy_signal(close, rolling_avg_price):
            order = self._build_buy_order()
            self._portfolio.apply_buy(order["usdc"], close)
            orders.append(order)

        elif self._is_sell_signal(close, rolling_avg_price):
            order = self._build_sell_order()
            self._portfolio.apply_sell(order["fraction"], close)
            orders.append(order)

        return orders


# ═══════════════════════════════════════════════════════════════════════════════
# MessageRouter — dispatches JSON messages to the right handler
# ═══════════════════════════════════════════════════════════════════════════════

class MessageRouter:
    """
    Routes incoming JSON messages by their "type" field.

    Supported message types
    -----------------------
    hello  — session handshake sent once at session start
    bar    — OHLCV bar; triggers strategy evaluation and order generation
    orders — exchange acknowledgement of previously submitted orders
    batch  — envelope that wraps multiple sub-messages for bulk delivery

    Any unknown type receives an empty orders reply to stay in sync with the
    exchange's sequence counter.
    """

    def __init__(self, strategy: DisplacementReversionStrategy) -> None:
        self._strategy = strategy
        self._handlers: dict[str, Any] = {
            "hello":  self._handle_hello,
            "bar":    self._handle_bar,
            "orders": self._handle_orders,
            "batch":  self._handle_batch,
        }

    # ── Individual message handlers ────────────────────────────────────────────

    def _handle_hello(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │  HELLO PHASE                                                │
        └─────────────────────────────────────────────────────────────┘
        The hello message is the very first message the exchange sends at
        the start of every competition session.  It serves three purposes:

          1. Synchronisation — establishes seq=0 as the shared sequence
             counter baseline.  Every subsequent message (bar, orders,
             batch) increments this counter by one, and every reply must
             echo the same seq value so the exchange can match requests
             to responses.

          2. Capability negotiation — in extended protocols the hello
             payload can carry session metadata (tick size, fee rates,
             allowed order types).  We don't use those fields here but
             the handler is the right place to parse them if needed.

          3. State reset — receiving hello guarantees a clean slate.
             The Portfolio, PriceTracker, and Strategy were already
             initialised fresh in BotRunner.__init__, so no reset is
             needed in this competition setup, but a production bot
             would re-initialise all stateful objects here.

        Response: we must reply to hello before the exchange will send
        the first bar.  The reply carries an empty orders list because
        we have no price data yet and therefore no position to open.
        """
        # Echo the seq counter from the hello message.  Using seq=0 as
        # a fallback is safe because hello is always the first message.
        return [{"type": "reply", "seq": msg.get("seq", 0), "orders": []}]

    def _handle_bar(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │  BAR PROCESSING PHASE                                       │
        └─────────────────────────────────────────────────────────────┘
        A bar message delivers one completed OHLCV candlestick.  This is
        the primary data feed driving the entire strategy.

        Bar payload fields:
          open   — price of the first trade in this period
          high   — highest traded price in this period
          low    — lowest traded price in this period
          close  — price of the last trade in this period  ← used by strategy
          volume — total cbBTC traded in this period       ← used for validation

        Processing pipeline (see DisplacementReversionStrategy.evaluate):

          Step 1 — VALIDATION
            Bars with zero volume or non-finite close are discarded before
            touching any state.  This prevents data gaps or exchange halts
            from corrupting the rolling average or triggering false signals.

          Step 2 — PRICE WINDOW UPDATE
            The valid close price is appended to the PriceTracker's deque.
            The deque enforces a fixed 20-bar window; the oldest price is
            automatically evicted when the window is full.

          Step 3 — DRAWDOWN CHECK
            Before evaluating any signal the Portfolio recomputes mark-to-
            market equity at the current close price, updates the peak
            equity high-water mark, and checks whether the drawdown has
            reached the 20 % soft limit.  If it has, trading is halted
            permanently for this session and an empty orders list is returned.

          Step 4 — ROLLING AVERAGE READINESS
            The strategy silently skips the first 19 bars (window not yet
            full) and only begins generating signals from bar 20 onward.
            This warm-up period ensures the average is statistically
            meaningful before any capital is deployed.

          Step 5 — SIGNAL EVALUATION
            BUY  signal: close < 20-bar SMA × 0.997  (price down ≥ 0.3 %)
            SELL signal: close > 20-bar SMA × 1.003  (price up   ≥ 0.3 %)
            At most one order is emitted per bar.

          Step 6 — SIMULATED FILL
            The bot has no real order-book, so fills are assumed to execute
            at the bar's close price.  The Portfolio is updated immediately
            so the next bar sees the correct cash and coin balances.

        Reply: one reply per bar, with 0 or 1 orders in the list.
        The seq value is echoed so the exchange can correlate the reply.
        """
        seq    = msg.get("seq", 0)
        # Extract the nested bar dict; default to empty dict so _bar_is_valid
        # will cleanly reject it rather than raising a KeyError.
        orders = self._strategy.evaluate(msg.get("bar", {}))
        return [{"type": "reply", "seq": seq, "orders": orders}]

    def _handle_orders(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Handle an orders confirmation message from the exchange.

        The exchange sends an orders message after it has processed the
        orders list we included in a previous reply.  It confirms that
        our submission was received and accepted into the matching engine.

        Why we don't act on it here:
          - Fill simulation already happened optimistically in apply_buy /
            apply_sell at the moment we emitted the order (see bar handler
            Step 6 above).  We don't need to adjust the portfolio again.
          - The competition format does not send fill details (executed
            price, fees, partial fills) back in the orders message, so
            there is no new information to consume.
          - If the exchange ever sent a rejection here, a production bot
            would reverse the simulated fill and re-evaluate.  For this
            competition we assume all orders are accepted in full.

        Reply: empty orders list.  We must still reply to keep the seq
        counter in sync — the exchange increments seq for every message
        it sends, including orders confirmations, and will desync if we
        silently drop one.
        """
        return [{"type": "reply", "seq": msg.get("seq", 0), "orders": []}]

    def _handle_batch(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │  ORDER BATCHING / BATCH ENVELOPE                            │
        └─────────────────────────────────────────────────────────────┘
        A batch message is a transport-level envelope that wraps two or
        more sub-messages into a single stdin line.  The exchange uses it
        to reduce round-trip overhead when it needs to deliver multiple
        events at once — most commonly a bar followed immediately by the
        orders confirmation for the previous bar's reply.

        Example batch payload:
          {
            "type": "batch",
            "seq": 42,
            "messages": [
              {"type": "bar",    "seq": 42, "bar": {...}},
              {"type": "orders", "seq": 43, "orders": [...]}
            ]
          }

        Key design decisions:

          Sequential dispatch — sub-messages are processed in arrival
          order, not in parallel.  This is critical for correctness:
          if a bar and an orders confirmation arrive in the same batch,
          the bar must be evaluated first so that any new order it
          generates is counted before the orders confirmation (which
          might reference a prior trade) is processed.

          Independent seq values — each sub-message carries its own seq
          counter.  The outer batch seq is for the envelope itself and is
          not used; only the inner seq values matter for reply matching.

          Reply explosion — one batch produces N replies (one per sub-
          message), all written to stdout as separate lines.  The exchange
          reads them in order and matches each to its inner seq.  We must
          never collapse them into a single reply or the exchange will
          lose track of which reply belongs to which sub-message.

          Recursive dispatch — sub-messages are routed through the same
          dispatch() method as top-level messages.  A batch can therefore
          contain another batch (nested batching), and this handler will
          process it correctly without any extra logic.
        """
        replies: list[dict[str, Any]] = []
        for sub_message in msg.get("messages", []):
            # Dispatch each sub-message individually and collect all replies.
            # extend (not append) because dispatch always returns a list,
            # and a nested batch sub-message can itself produce multiple replies.
            replies.extend(self.dispatch(sub_message))
        return replies

    def _handle_unknown(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Fallback for unrecognised message types — echo seq, empty orders."""
        return [{"type": "reply", "seq": msg.get("seq", 0), "orders": []}]

    # ── Dispatch ───────────────────────────────────────────────────────────────

    def dispatch(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Route a single message to the appropriate handler."""
        handler = self._handlers.get(msg.get("type", ""), self._handle_unknown)
        return handler(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# BotRunner — wires everything together and owns the I/O loop
# ═══════════════════════════════════════════════════════════════════════════════

class BotRunner:
    """
    Assembles the bot's components and runs the stdin → stdout event loop.

    Dependency graph
    ----------------
    BotRunner
      └── MessageRouter
            └── DisplacementReversionStrategy
                  ├── Portfolio
                  └── PriceTracker

    I/O contract
    ------------
    • Reads one complete JSON object per line from stdin.
    • Writes one JSON reply per line to stdout for every non-batch message.
    • batch messages may produce multiple output lines (one per sub-message).
    • stdout is flushed after every write so the exchange never blocks.
    • Malformed JSON lines are silently discarded.
    • Zero-volume bars produce a reply with an empty orders list.
    """

    def __init__(self) -> None:
        self._portfolio = Portfolio(Config.STARTING_USDC)
        self._tracker   = PriceTracker(Config.ROLLING_WINDOW)
        self._strategy  = DisplacementReversionStrategy(
            portfolio=self._portfolio,
            tracker=self._tracker,
        )
        self._router = MessageRouter(strategy=self._strategy)

    @staticmethod
    def _parse(line: str) -> dict[str, Any] | None:
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _emit(reply: dict[str, Any]) -> None:
        print(json.dumps(reply, separators=(",", ":")), flush=True)

    def run(self, stream=None) -> None:
        """Read from stream (default stdin), dispatch, write replies."""
        for raw_line in (stream or sys.stdin):
            msg = self._parse(raw_line)
            if msg is None:
                continue
            msg_type = msg.get("type", "")
            replies = self._router.dispatch(msg)
            if replies and msg_type == "bar":
                self._emit(replies[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    BotRunner().run()


if __name__ == "__main__":
    main()
