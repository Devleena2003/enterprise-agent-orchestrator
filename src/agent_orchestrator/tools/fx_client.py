import logging
import re
import time
import httpx
from pydantic import BaseModel

from agent_orchestrator.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Approximate static USD-anchored fallback rates used ONLY if live FX API is unreachable
_APPROX_USD_PER_UNIT: dict[str, float] = {
    "USD": 1.0,
    "INR": 0.0119,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0066,
}


class FxConversionResult(BaseModel):
    tool: str = "external_fx"
    amount: float
    base_currency: str
    target_currency: str
    rate: float
    converted_amount: float
    rate_date: str
    is_fallback: bool
    source: str
    warning: str | None = None


def extract_amount_from_query(query: str) -> float | None:
    """Extract a standalone numeric amount for direct FX conversion queries."""
    cleaned = query.replace(",", "")
    # Match numbers not attached to hyphens (skip dates like 01-08-23)
    matches = re.findall(r"(?<![-\d/])\b(\d+(?:\.\d+)?)\b(?![-\d/])", cleaned)
    for m in matches:
        val = float(m)
        if val > 0:
            return val
    return None


class FxClient:
    """Resilient FX converter using free no-key Frankfurter API with static fallback."""

    def __init__(
        self,
        base_url: str = "https://api.frankfurter.app",
        timeout_seconds: float = 5.0,
        max_retries: int = 1,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)
        self._max_retries = max_retries
        self._transport = transport

    def convert(
        self, *, amount: float, base_currency: str, target_currency: str
    ) -> FxConversionResult:
        base = base_currency.strip().upper()
        target = target_currency.strip().upper()

        if base == target:
            return FxConversionResult(
                amount=amount,
                base_currency=base,
                target_currency=target,
                rate=1.0,
                converted_amount=round(amount, 2),
                rate_date="same_currency",
                is_fallback=False,
                source="identity",
            )

        last_err = "unknown_error"
        for attempt in range(self._max_retries + 1):
            try:
                with httpx.Client(
                    base_url=self._base_url,
                    timeout=self._timeout,
                    transport=self._transport,
                ) as client:
                    resp = client.get(
                        "/latest",
                        params={"amount": amount, "from": base, "to": target},
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    converted = float(data["rates"][target])
                    rate = converted / amount if amount else 0.0
                    return FxConversionResult(
                        amount=amount,
                        base_currency=base,
                        target_currency=target,
                        rate=round(rate, 6),
                        converted_amount=round(converted, 2),
                        rate_date=str(data.get("date", "latest")),
                        is_fallback=False,
                        source="frankfurter_live_api",
                    )
                last_err = f"HTTP {resp.status_code}"
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"

            if attempt < self._max_retries:
                time.sleep(0.2 * (2**attempt))

        logger.warning(
            "Live FX API failed (%s); executing static fallback for %s->%s.",
            last_err,
            base,
            target,
        )
        return self._fallback_convert(amount=amount, base=base, target=target, reason=last_err)

    def _fallback_convert(
        self, *, amount: float, base: str, target: str, reason: str
    ) -> FxConversionResult:
        if base in _APPROX_USD_PER_UNIT and target in _APPROX_USD_PER_UNIT:
            rate = _APPROX_USD_PER_UNIT[base] / _APPROX_USD_PER_UNIT[target]
            return FxConversionResult(
                amount=amount,
                base_currency=base,
                target_currency=target,
                rate=round(rate, 6),
                converted_amount=round(amount * rate, 2),
                rate_date="static_fallback",
                is_fallback=True,
                source="static_fallback_approximate",
                warning=f"Live FX API unavailable ({reason}); used approximate static fallback rate.",
            )
        return FxConversionResult(
            amount=amount,
            base_currency=base,
            target_currency=target,
            rate=0.0,
            converted_amount=0.0,
            rate_date="unavailable",
            is_fallback=True,
            source="unavailable",
            warning=f"Live FX API unavailable ({reason}) and no fallback rate for {base}->{target}.",
        )


def get_fx_client(settings: Settings | None = None) -> FxClient:
    cfg = settings or get_settings()
    return FxClient(
        base_url=cfg.fx_api_base_url,
        timeout_seconds=cfg.fx_timeout_seconds,
    )