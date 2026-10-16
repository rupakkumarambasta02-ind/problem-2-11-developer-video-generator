"""Single source of truth for number/identifier tokenization and matching (spec §7.1, §7.2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from typing import Iterable, Literal

NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?")
_TOKEN_RE = re.compile(r"(?:[A-Za-z0-9_@.\-/]|(?<=\d),(?=\d{3}(?!\d)))+%?")

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_SCALES = {"thousand": 1_000, "million": 1_000_000}
_NUMWORD = "|".join(sorted([*_UNITS, *_TENS, "hundred", *_SCALES], key=len, reverse=True))
_SEQ_RE = re.compile(
    rf"\b(?:{_NUMWORD})(?:(?:\s+and\s+|\s+|-)(?:{_NUMWORD}))*\b", re.IGNORECASE)


@dataclass(frozen=True)
class Claim:
    kind: Literal["number", "identifier"]
    token: str


def normalize_text(text: str) -> str:
    return text.replace("−", "-")


_AND_RE = re.compile(r"\band\b")


def _word_kind(word: str) -> str:
    if word == "hundred":
        return "hundred"
    if word in _SCALES:
        return "scale"
    if word in _TENS:
        return "tens"
    return "teen" if _UNITS[word] >= 10 else "unit"


class _Number:
    """One spoken number being built from a run of number words."""

    def __init__(self, word: str, kind: str) -> None:
        self.total = 0
        self.group = 0
        self.has_hundred = False
        self.last = kind
        self._add(word, kind)

    def value(self) -> int:
        return self.total + self.group

    def _add(self, word: str, kind: str) -> None:
        if kind == "hundred":
            self.group = (self.group or 1) * 100
            self.has_hundred = True
        elif kind == "scale":
            self.total += (self.group or 1) * _SCALES[word]
            self.group = 0
            self.has_hundred = False
        else:
            self.group += _UNITS.get(word, _TENS.get(word, 0))
        self.last = kind

    def continues(self, word: str, kind: str, joined_by_and: bool) -> bool:
        """Whether `word` extends this number rather than starting a new one."""
        last = self.last
        if joined_by_and:
            # "and" only joins after hundred/thousand/million, before a unit, teen or tens word.
            return last in ("hundred", "scale") and kind in ("unit", "teen", "tens")
        if kind in ("unit", "teen"):
            if last in ("hundred", "scale"):
                return True
            # Tens followed by a unit 1-9 (twenty-one); a teen or a zero never follows tens.
            return last == "tens" and kind == "unit" and _UNITS[word] >= 1
        if kind == "tens":
            return last in ("hundred", "scale")
        if kind == "hundred":
            return last == "unit" and not self.has_hundred
        if kind == "scale":
            return last in ("unit", "teen", "tens", "hundred")
        return False

    def apply(self, word: str, kind: str) -> None:
        self._add(word, kind)


def words_to_digits(text: str) -> str:
    def repl(m: re.Match) -> str:
        run = m.group(0)
        out: list[str] = []
        current: _Number | None = None
        prev_end = 0
        for word_m in re.finditer(r"[A-Za-z]+", run):
            word = word_m.group(0).lower()
            if word == "and":
                continue  # "and" stays in the separator text
            separator = run[prev_end:word_m.start()]
            prev_end = word_m.end()
            kind = _word_kind(word)
            joined_by_and = bool(_AND_RE.search(separator))
            if current is not None and current.continues(word, kind, joined_by_and):
                current.apply(word, kind)
                continue
            if current is not None:
                out.append(str(current.value()))
                out.append(separator)
            current = _Number(word, kind)
        if current is not None:
            out.append(str(current.value()))
        return "".join(out)
    return _SEQ_RE.sub(repl, text)


def split_tokens(text: str) -> list[str]:
    out = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).rstrip(".-/")
        if tok:
            out.append(tok)
    return out


def classify(token: str) -> Literal["number", "identifier"] | None:
    if NUMBER_RE.fullmatch(token):
        return "number"
    if any(ch.isdigit() for ch in token):
        return "identifier"
    return None


def _prepared_tokens(text: str) -> list[str]:
    return split_tokens(words_to_digits(normalize_text(text)))


def extract_claims(text: str) -> list[Claim]:
    claims = []
    for tok in _prepared_tokens(text):
        kind = classify(tok)
        if kind:
            claims.append(Claim(kind, tok))
    return claims


def source_tokens(text: str) -> set[str]:
    return set(_prepared_tokens(text))


def source_number_literals(text: str) -> list[str]:
    return [t for t in _prepared_tokens(text) if classify(t) == "number"]


_MODES = (ROUND_HALF_UP, ROUND_HALF_EVEN)


def _parse(literal: str) -> tuple[Decimal, bool, int] | None:
    pct = literal.endswith("%")
    core = (literal[:-1] if pct else literal).replace(",", "")
    try:
        value = Decimal(core)
    except InvalidOperation:
        return None
    dec = len(core.split(".", 1)[1]) if "." in core else 0
    return value, pct, dec


def decimals(literal: str) -> int:
    parsed = _parse(literal)
    return parsed[2] if parsed else 0


def _sig_figs(d: Decimal) -> int:
    return len("".join(str(x) for x in d.as_tuple().digits).lstrip("0"))


def _rounded(d: Decimal, n: int, mode: str) -> Decimal:
    return d.quantize(Decimal(1).scaleb(-n), rounding=mode)


def _rounds_to(v: Decimal, vdec: int, s: Decimal, max_n: int) -> bool:
    for n in range(0, max_n + 1):
        if n != vdec:
            continue
        for mode in _MODES:
            r = _rounded(s, n, mode)
            if r == v and _sig_figs(r) >= 2:
                return True
    return False


def _literal_matches(v: Decimal, v_pct: bool, v_dec: int, lit: str) -> bool:
    ps = _parse(lit)
    if ps is None:
        return False
    s, s_pct, s_dec = ps
    if v_pct == s_pct:
        if v == s and v_dec <= s_dec:
            return True
        return s_dec > 0 and _rounds_to(v, v_dec, s, s_dec - 1)
    if v_pct and not s_pct:
        return _rounds_to(v, v_dec, s * 100, max(s_dec - 2, 0))
    return False


def number_grounded(value: str, literals: Iterable[str]) -> bool:
    pv = _parse(value)
    if pv is None:
        return False
    v, v_pct, v_dec = pv
    for lit in literals:
        # Source text is untrusted: size the context to the operands so quantize and
        # the percent multiply never hit the default 28-digit limit, and treat any
        # remaining InvalidOperation as no match for this literal.
        with localcontext() as ctx:
            ctx.prec = max(28, len(value) + len(lit) + 10)
            try:
                if _literal_matches(v, v_pct, v_dec, lit):
                    return True
            except InvalidOperation:
                continue
    return False


def identifier_grounded(token: str, source_texts: Iterable[str]) -> bool:
    return any(token in source_tokens(t) for t in source_texts)
