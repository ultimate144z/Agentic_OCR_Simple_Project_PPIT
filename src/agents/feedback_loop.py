"""
Feedback Loop — Self-Evaluation & Quality Assessment
=====================================================
The Feedback Loop evaluates the quality of OCR output and triggers
retry/improvement cycles when the result is below acceptable thresholds.

It also supports learning from user corrections — when the user edits
the OCR text before saving, the Feedback Loop detects what changed and
stores those corrections in memory for future runs.

Phase 2 Requirement Mapping:
  - Slide 25 (Operational Workflow: Learn)
  - Slide 23 (Agent Architecture: Feedback)
  - Slide 27 (Memory & Context: learning from corrections)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from src.agents.memory_store import MemoryStore
from src.agents.agent_logger import AgentLogger


@dataclass
class QualityReport:
    """Result of an OCR quality evaluation."""
    overall_score: float = 0.0         # 0.0 – 1.0
    avg_confidence: float = 0.0        # average OCR confidence
    word_count: int = 0
    line_count: int = 0
    gibberish_ratio: float = 0.0       # ratio of non-word tokens
    short_word_ratio: float = 0.0      # ratio of very short (1-2 char) words
    is_acceptable: bool = True         # meets the confidence threshold
    issues: List[str] = field(default_factory=list)
    suggestion: str = ""               # what to do next


class FeedbackLoop:
    """
    Evaluates OCR output quality and learns from user corrections.

    Phase 1 had NO self-assessment — the OCR output was blindly accepted
    regardless of quality. The agentic version critically evaluates its
    own output and takes action when quality is poor.
    """

    def __init__(self, memory: MemoryStore, logger: AgentLogger) -> None:
        self.memory = memory
        self.logger = logger

    def evaluate_quality(
        self,
        text: str,
        confidence_values: Optional[List[float]] = None,
        threshold: float = 0.5,
    ) -> QualityReport:
        """
        Evaluate the quality of OCR output.

        Parameters
        ----------
        text : str
            The extracted OCR text.
        confidence_values : list of float, optional
            Per-word confidence scores from the OCR engine.
        threshold : float
            Minimum acceptable overall quality score.
        """
        report = QualityReport()

        if not text or not text.strip():
            report.overall_score = 0.0
            report.is_acceptable = False
            report.issues.append("OCR produced empty output")
            report.suggestion = "Retry with a different OCR engine or preprocessing"
            return report

        # ---- Basic stats ----
        lines = [l for l in text.strip().splitlines() if l.strip()]
        words = text.split()
        report.line_count = len(lines)
        report.word_count = len(words)

        # ---- Confidence score ----
        if confidence_values and len(confidence_values) > 0:
            report.avg_confidence = sum(confidence_values) / len(confidence_values)
        else:
            report.avg_confidence = self._estimate_confidence_from_text(text)

        # ---- Gibberish detection ----
        report.gibberish_ratio = self._compute_gibberish_ratio(words)

        # ---- Short word ratio ----
        short_words = [w for w in words if len(w) <= 2 and w.isalpha()]
        report.short_word_ratio = len(short_words) / max(1, len(words))

        # ---- Fragmentation detection (single-char token ratio) ----
        # When OCR reads handwriting character-by-character, most tokens are
        # length-1 (e.g. "L", "8", "H"). This is the strongest signal of failure.
        single_char_tokens = [w for w in words if len(w) == 1]
        single_char_ratio = len(single_char_tokens) / max(1, len(words))

        # ---- Numeric dominance ----
        numeric_tokens = [w for w in words if re.match(r'^\d+$', w)]
        numeric_ratio = len(numeric_tokens) / max(1, len(words))

        # ---- Average token length (short = fragmented) ----
        avg_word_len = sum(len(w) for w in words) / max(1, len(words))

        # ---- Compute overall score ----
        conf_score = report.avg_confidence
        gibberish_penalty = report.gibberish_ratio * 0.5
        short_penalty = max(0, report.short_word_ratio - 0.3) * 0.2
        # Reduced: Tesseract on handwriting naturally produces single-char tokens
        # — penalise but don't treat it as total failure
        fragmentation_penalty = single_char_ratio * 0.65
        # Excessive isolated digits signals diagram/table confusion
        numeric_penalty = max(0, numeric_ratio - 0.2) * 0.3
        # Very low average token length = fragmented output
        short_len_penalty = max(0, (3.0 - avg_word_len) / 3.0) * 0.15
        word_bonus = min(0.15, report.word_count / 500.0)

        report.overall_score = max(
            0.0,
            min(1.0, conf_score - gibberish_penalty - short_penalty
                - fragmentation_penalty - numeric_penalty - short_len_penalty
                + word_bonus),
        )

        # ---- Identify issues ----
        if report.avg_confidence < 0.4:
            report.issues.append(f"Low OCR confidence ({report.avg_confidence:.0%})")
        if report.gibberish_ratio > 0.3:
            report.issues.append(f"High gibberish ratio ({report.gibberish_ratio:.0%})")
        if report.word_count < 5:
            report.issues.append(f"Very few words extracted ({report.word_count})")
        if report.short_word_ratio > 0.5:
            report.issues.append("Many single/double-character words (fragmented OCR)")
        if single_char_ratio > 0.3:
            report.issues.append(
                f"Highly fragmented output — {single_char_ratio:.0%} single-character tokens "
                f"(OCR treating each stroke as a separate character)"
            )
        if numeric_ratio > 0.4:
            report.issues.append(
                f"Numeric token dominance ({numeric_ratio:.0%}) — "
                f"engine likely confused by diagrams or formulas"
            )

        # ---- Acceptability ----
        report.is_acceptable = report.overall_score >= threshold

        if not report.is_acceptable:
            report.suggestion = "Quality below threshold — retry with different engine/preprocessing"
        elif report.issues:
            report.suggestion = "Quality acceptable but with issues — review recommended"
        else:
            report.suggestion = "Quality looks good"

        # ---- Log the evaluation ----
        self.logger.log_decision(
            agent="FeedbackLoop",
            action=f"Quality evaluation: {report.overall_score:.2f} ({'PASS' if report.is_acceptable else 'FAIL'})",
            reasoning=f"confidence={report.avg_confidence:.2f}, gibberish={report.gibberish_ratio:.2f}, words={report.word_count}",
            confidence=report.overall_score,
            outcome="acceptable" if report.is_acceptable else "needs_retry",
            metadata={
                "issues": report.issues,
                "suggestion": report.suggestion,
            },
        )

        return report

    def learn_from_corrections(self, original_text: str, corrected_text: str) -> int:
        """
        Compare original OCR output with user-corrected version.
        Extract correction patterns and store in memory for future use.

        Returns the number of corrections learned.
        """
        if not original_text or not corrected_text:
            return 0

        original_words = original_text.split()
        corrected_words = corrected_text.split()

        corrections_learned = 0
        matcher = SequenceMatcher(None, original_words, corrected_words)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                # User replaced words — learn this pattern
                orig_chunk = " ".join(original_words[i1:i2])
                corr_chunk = " ".join(corrected_words[j1:j2])

                if orig_chunk.lower() != corr_chunk.lower():
                    self.memory.learn_correction(orig_chunk, corr_chunk)
                    corrections_learned += 1

        if corrections_learned > 0:
            self.logger.log_decision(
                agent="FeedbackLoop",
                action=f"Learned {corrections_learned} correction(s) from user edits",
                reasoning="User edited the OCR output before saving — patterns stored for future runs",
                confidence=1.0,
                outcome=f"{corrections_learned} new correction patterns saved to memory",
            )

        return corrections_learned

    def apply_learned_corrections(self, text: str) -> Tuple[str, int]:
        """
        Apply previously learned correction patterns to new OCR text.
        Returns the corrected text and the number of corrections applied.
        """
        corrections = self.memory.get_corrections()
        if not corrections:
            return text, 0

        result = text
        applied = 0

        for wrong, correct in corrections.items():
            if wrong.lower() in result.lower():
                # Case-insensitive replacement
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                new_result = pattern.sub(correct, result)
                if new_result != result:
                    result = new_result
                    applied += 1

        if applied > 0:
            self.logger.log_decision(
                agent="FeedbackLoop",
                action=f"Applied {applied} learned correction(s) to OCR output",
                reasoning="Corrections from previous user edits were applied automatically",
                confidence=0.9,
                outcome=f"{applied} corrections applied",
            )

        return result, applied

    def _estimate_confidence_from_text(self, text: str) -> float:
        """
        When no per-word confidence is available, estimate from text characteristics.
        """
        words = text.split()
        if not words:
            return 0.0

        # Heuristic: ratio of recognizable words (>2 chars, mostly alpha)
        good_words = sum(
            1 for w in words
            if len(w) > 2 and sum(c.isalpha() for c in w) / max(1, len(w)) > 0.6
        )

        return good_words / len(words)

    def _compute_gibberish_ratio(self, words: List[str]) -> float:
        """
        Estimate the ratio of gibberish tokens in the text.
        Gibberish = tokens that are unlikely to be real words.
        """
        if not words:
            return 0.0

        gibberish_count = 0
        for word in words:
            clean = re.sub(r"[^a-zA-Z]", "", word)
            if not clean:
                continue

            # Heuristics for gibberish:
            # 1. Too many consonants in a row
            if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", clean.lower()):
                gibberish_count += 1
                continue

            # 2. Very low vowel ratio for longer words
            if len(clean) > 4:
                vowel_count = sum(1 for c in clean.lower() if c in "aeiou")
                if vowel_count / len(clean) < 0.15:
                    gibberish_count += 1
                    continue

        alpha_words = [w for w in words if any(c.isalpha() for c in w)]
        return gibberish_count / max(1, len(alpha_words))
