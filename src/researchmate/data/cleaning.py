"""Text cleaning and artifact removal for QASPER papers and annotations.

All cleaning operations are deterministic and idempotent:
    clean_text(clean_text(x)) == clean_text(x)

Raw text is preserved alongside cleaned text in the Paragraph model
so that ground-truth evidence matching can always reference original spans.
"""

from __future__ import annotations

import re

from researchmate.data.models import Paper, Paragraph

# Pre-compiled regex patterns for artifact removal and normalization
_RE_FLOAT_SELECTED = re.compile(r"\bFLOAT\s+SELECTED\b\s*:?", re.IGNORECASE)
_RE_BIBREF = re.compile(r"\b[A-Za-z]*BIBREF\d*\b")
_RE_STRUCT_REF = re.compile(r"\b(?:SEC|TAB|FIG|EQ|UNK)REF\d*\b")
_RE_FORM_PLACEHOLDER = re.compile(r"\b(?:INLINE|DISPLAY)FORM\d*\b")

# LaTeX formatting and math patterns
_RE_DISPLAY_MATH = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_RE_INLINE_MATH = re.compile(r"\$([^$\n]+)\$")
_RE_LATEX_TEXT_CMD = re.compile(
    r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|mathsf|mathtt|mathcal|emph)\{([^{}]+)\}"
)
_RE_LATEX_FONT_SWITCH_BRACED = re.compile(r"\{\\(?:rm|bf|it|cal|sf|tt)\s+([^{}]+)\}")
_RE_LATEX_FONT_SWITCH_BARE = re.compile(r"\\(?:rm|bf|it|cal|sf|tt)\b")
_RE_LATEX_ESCAPED = re.compile(r"\\([%&_#])")
_RE_LATEX_STRAY_CMD = re.compile(r"\\[a-zA-Z]+\b")

# Known LaTeX symbol replacements (with surrounding spaces to preserve separation)
_LATEX_SYMBOLS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\\(?:cdot|cdots|ldots|dots)\b"), " ... "),
    (re.compile(r"\\(?:times)\b"), " x "),
    (re.compile(r"\\(?:pm|mp)\b"), " +/- "),
    (re.compile(r"\\(?:le|leq)\b"), " <= "),
    (re.compile(r"\\(?:ge|geq)\b"), " >= "),
    (re.compile(r"\\(?:neq|ne)\b"), " != "),
    (re.compile(r"\\(?:approx|sim)\b"), " ~ "),
]

# Orphaned brackets & citation prepositions left after token removal
_RE_EMPTY_ORPHAN_PARENS = re.compile(
    r"\(\s*(?:(?:\b(?:see|also|cf)\b|e\.g\.|i\.e\.)\s*[,;.]*|\s|[,;.])*\)",
    re.IGNORECASE,
)
_RE_EMPTY_ORPHAN_BRACKETS = re.compile(
    r"\[\s*(?:(?:\b(?:see|also|cf)\b|e\.g\.|i\.e\.)\s*[,;.]*|\s|[,;.])*\]",
    re.IGNORECASE,
)
_RE_EMPTY_BRACES = re.compile(r"\{\s*[,;.]*\s*\}")

# Punctuation & whitespace patterns
_RE_HANGING_PARENS_START = re.compile(r"\(\s*[,;]\s*")
_RE_HANGING_PARENS_END = re.compile(r"\s*[,;]\s*\)")
_RE_HANGING_BRACKETS_START = re.compile(r"\[\s*[,;]\s*")
_RE_HANGING_BRACKETS_END = re.compile(r"\s*[,;]\s*\]")
_RE_DUPLICATE_COMMAS_SEMICOLONS = re.compile(r"([,;])\s*[,;]+")
_RE_MULTI_DOTS = re.compile(r"\.{4,}")
_RE_DOUBLE_DOT = re.compile(r"(?<!\.)\.\.(?!\.)")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,;:]|(?<!=)!(?!=)|\?(?!\?)|(?<!\.)\.(?!\.))")
_RE_WHITESPACE = re.compile(r"[\r\n\t]+")
_RE_MULTI_SPACE = re.compile(r" {2,}")


def clean_text(text: str) -> str:
    """Clean raw text from QASPER dataset artifacts while preserving semantics.

    Handles:
    - ``FLOAT SELECTED`` table/figure evidence prefixes
    - ``BIBREF`` citation tokens (e.g. ``BIBREF0``, ``BERTBIBREF1``)
    - Structural reference tokens (``TABREF``, ``FIGREF``, ``SECREF``, ``EQREF``, ``UNKREF``)
    - Formula placeholders (``INLINEFORM``, ``DISPLAYFORM``)
    - LaTeX formatting, math delimiters, font tags, and stray backslashes
    - Orphaned brackets and punctuation left after token removal
    - Whitespace normalization (newlines, tabs, consecutive spaces)

    Guaranteed to be idempotent:
        ``clean_text(clean_text(x)) == clean_text(x)``

    Args:
        text: Raw text string to clean.

    Returns:
        Cleaned, normalized text string.
    """
    if not text:
        return ""

    # 1. Protect escaped literal dollars (\$) from being treated as math delimiters
    text = text.replace(r"\$", "__LITERAL_DOLLAR__")

    # 2. Remove FLOAT SELECTED prefix
    text = _RE_FLOAT_SELECTED.sub("", text)

    # 3. Remove citation tokens
    text = _RE_BIBREF.sub("", text)

    # 4. Remove structural reference placeholders
    text = _RE_STRUCT_REF.sub("", text)

    # 5. Remove formula placeholders
    text = _RE_FORM_PLACEHOLDER.sub("", text)

    # 6. LaTeX cleanup
    # Unwrap display math $$...$$ and inline math $...$
    text = _RE_DISPLAY_MATH.sub(r" \1 ", text)
    text = _RE_INLINE_MATH.sub(r" \1 ", text)

    # Restore literal dollars
    text = text.replace("__LITERAL_DOLLAR__", "$")

    # Unwrap LaTeX font/text formatting commands with curly braces (e.g. \textbf{xyz})
    for _ in range(3):
        text = _RE_LATEX_TEXT_CMD.sub(r"\1", text)

    # Bare font switches (e.g. {\rm Encoder} or \rm)
    text = _RE_LATEX_FONT_SWITCH_BRACED.sub(r"\1", text)
    text = _RE_LATEX_FONT_SWITCH_BARE.sub("", text)

    # Symbol replacements
    for pat, rep in _LATEX_SYMBOLS:
        text = pat.sub(rep, text)

    # Escaped LaTeX characters (\%, \&, \_, etc.)
    text = _RE_LATEX_ESCAPED.sub(r"\1", text)

    # Strip remaining LaTeX commands (e.g. \alpha, \sum)
    text = _RE_LATEX_STRAY_CMD.sub("", text)
    text = text.replace("\\", "")

    # 7. Clean orphaned brackets, parentheses, and braces left by removed tokens
    for _ in range(3):
        text = _RE_EMPTY_ORPHAN_PARENS.sub("", text)
        text = _RE_EMPTY_ORPHAN_BRACKETS.sub("", text)
        text = _RE_EMPTY_BRACES.sub("", text)
        text = _RE_HANGING_PARENS_START.sub("(", text)
        text = _RE_HANGING_PARENS_END.sub(")", text)
        text = _RE_HANGING_BRACKETS_START.sub("[", text)
        text = _RE_HANGING_BRACKETS_END.sub("]", text)

    # 8. Normalize repeated punctuation and spaces before punctuation
    text = _RE_DUPLICATE_COMMAS_SEMICOLONS.sub(r"\1", text)
    text = _RE_MULTI_DOTS.sub("...", text)
    text = _RE_DOUBLE_DOT.sub(".", text)
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)

    # 9. Normalize whitespace
    text = _RE_WHITESPACE.sub(" ", text)
    text = _RE_MULTI_SPACE.sub(" ", text)

    return text.strip()


def clean_paragraph(para: Paragraph) -> Paragraph:
    """Return a new Paragraph with cleaned text while preserving raw_text.

    Args:
        para: Original paragraph to clean.

    Returns:
        A new Paragraph instance with ``text=clean_text(...)`` and
        ``raw_text`` pointing to the original raw text.
    """
    original_raw = para.raw_text or para.text
    cleaned = clean_text(original_raw)
    # If cleaning results in empty string, fallback to non-empty fallback
    if not cleaned:
        cleaned = original_raw.strip()

    return Paragraph(
        paper_id=para.paper_id,
        section=para.section,
        index=para.index,
        text=cleaned,
        raw_text=original_raw,
    )


def clean_paper(paper: Paper) -> Paper:
    """Return a new Paper with all paragraphs cleaned.

    The original raw text is preserved in ``Paragraph.raw_text``.

    Args:
        paper: Original Paper instance.

    Returns:
        A new Paper instance with cleaned paragraphs.
    """
    cleaned_paras = [clean_paragraph(p) for p in paper.paragraphs]
    return Paper(
        paper_id=paper.paper_id,
        title=clean_text(paper.title) or paper.title,
        abstract=clean_text(paper.abstract) if paper.abstract else "",
        paragraphs=cleaned_paras,
    )


def clean_evidence(evidence: str) -> str:
    """Clean an evidence string, removing FLOAT SELECTED: prefix and artifacts.

    Args:
        evidence: Raw evidence string from QASPER annotations.

    Returns:
        Cleaned evidence string.
    """
    return clean_text(evidence)
