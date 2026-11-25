"""
Redaction for stored tool payloads and messages (docs/industry_norms_2026-11.md §1, decision D3).

Secrets are always redacted before storage. PII redaction is configurable:
  "off"           keep PII (the raw layer is the user's own database; the industry norm)
  "emails_phones" emails and phone numbers
  "strict"        also IP addresses, card numbers (Luhn-checked) and US SSNs

Redaction is applied to strings and recursively to JSON (dicts / lists). A value
stored under a sensitive key (password, token, api_key, authorization, ...) is
redacted whatever it looks like.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Tuple

PII_MODES = ("off", "emails_phones", "strict")

_SECRET_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}")),
    ("nvidia_key", re.compile(r"\bnvapi-[A-Za-z0-9_-]{20,}")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("url_credentials", re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)")),
    ("assigned_secret", re.compile(
        r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?key|secret[_-]?key|token|auth[_-]?token)"
        r"(\s*[=:]\s*)(['\"]?)([^\s'\",;]{4,})\3")),
]

_SENSITIVE_KEY = re.compile(
    r"(?i)^(password|passwd|pwd|secret|client_secret|api[_-]?key|apikey|access[_-]?key|secret[_-]?key|"
    r"token|access[_-]?token|refresh[_-]?token|auth[_-]?token|authorization|private[_-]?key|cookie|session[_-]?id)$")

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<![\w.])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]\d{3,4}[\s.-]\d{3,4}(?![\w.])")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def redact_text(text: str, pii: str = "off") -> str:
    """Redact secrets (always) and PII (per mode) in a string."""
    if not text:
        return text
    for name, pat in _SECRET_PATTERNS:
        if name == "assigned_secret":
            text = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}[REDACTED:secret]{m.group(3)}", text)
        else:
            text = pat.sub(f"[REDACTED:{name}]", text)
    if pii in ("emails_phones", "strict"):
        text = _EMAIL.sub("[REDACTED:email]", text)
        text = _PHONE.sub("[REDACTED:phone]", text)
    if pii == "strict":
        text = _SSN.sub("[REDACTED:ssn]", text)
        text = _IPV4.sub("[REDACTED:ip]", text)
        text = _CARD.sub(lambda m: "[REDACTED:card]" if _luhn(re.sub(r"\D", "", m.group(0))) else m.group(0), text)
    return text


def redact(value: Any, pii: str = "off") -> Any:
    """Redact a string or a JSON value (dicts / lists) recursively; other types pass through."""
    return _walk(value, lambda s: redact_text(s, pii))


def _walk(value: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and _SENSITIVE_KEY.match(k) and v not in (None, "", [], {}):
                out[k] = "[REDACTED:secret]"
            else:
                out[k] = _walk(v, fn)
        return out
    if isinstance(value, (list, tuple)):
        return [_walk(v, fn) for v in value]
    return value
