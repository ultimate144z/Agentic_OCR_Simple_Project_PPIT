"""
Decision Engine — Adaptive Strategy Selection
===============================================
The Decision Engine is the "brain" of the agentic system. It takes the
ImageProfile from the Perception Agent and decides:
  1. Which OCR engine to use
  2. What preprocessing strategy to apply
  3. Whether to use dual-channel (red ink) mode
  4. Whether to retry with a different strategy if quality is low

Unlike Phase 1 — where the engine was either hardcoded or user-selected,
and preprocessing was always the same — the Decision Engine adapts its
strategy based on perception data AND historical performance from memory.

Phase 2 Requirement Mapping:
  - Slide 22 (Agentic Vision: Tool → Agent, Reactive → Proactive)
  - Slide 24 (Agent Type Selection: Goal-based agent)
  - Slide 26 (Intelligence Layer: Rules + ML performance data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agents.perception_agent import ImageProfile
from src.agents.memory_store import MemoryStore
from src.agents.agent_logger import AgentLogger
from src.agents.safety_guard import SafetyGuard, SafetyCheck


@dataclass
class ProcessingStrategy:
    """The complete strategy for processing an image."""
    ocr_engine: str = "easyocr"             # "easyocr", "tesseract", "paddleocr", "trocr"
    preprocessing: str = "standard"          # "standard", "enhanced", "minimal", "aggressive"
    use_dual_channel: bool = False           # extract red ink separately
    apply_deskew: bool = False
    apply_upscale: bool = False
    apply_brightness_enhance: bool = False
    apply_contrast_enhance: bool = False
    apply_sharpening: bool = False
    confidence_threshold: float = 0.5        # minimum acceptable OCR confidence
    max_retries: int = 2                     # how many retry attempts with different engines
    reasoning: List[str] = field(default_factory=list)  # why this strategy was chosen


class DecisionEngine:
    """
    Goal-Based Agent: selects the optimal processing strategy for each image.

    The agent's GOAL is to maximize OCR accuracy. It pursues this goal by:
      1. Reading the ImageProfile (perception data)
      2. Consulting memory (what worked before for similar images)
      3. Applying decision rules to select the best strategy
      4. Logging every decision for transparency

    This is a Goal-Based Agent (Slide 24) because it evaluates multiple
    strategies against the goal of maximum accuracy, rather than just
    reacting to a stimulus.
    """

    def __init__(
        self,
        memory: MemoryStore,
        logger: AgentLogger,
        safety: SafetyGuard,
    ) -> None:
        self.memory = memory
        self.logger = logger
        self.safety = safety

    def decide_strategy(self, profile: ImageProfile) -> ProcessingStrategy:
        """
        Given an ImageProfile, decide the optimal processing strategy.

        This is the core intelligence of the agent — replacing the static
        Phase 1 pipeline with adaptive decision-making.
        """
        strategy = ProcessingStrategy()
        strategy.reasoning = []

        # ---- Step 1: Select OCR Engine ----
        strategy.ocr_engine = self._select_engine(profile, strategy)

        # ---- Step 2: Determine Preprocessing ----
        self._decide_preprocessing(profile, strategy)

        # ---- Step 3: Dual Channel Decision ----
        self._decide_dual_channel(profile, strategy)

        # ---- Step 4: Set Confidence Threshold from memory ----
        threshold = self.memory.get_preference("confidence_threshold", 0.5)
        strategy.confidence_threshold = threshold

        # ---- Step 5: Determine Retry Count ----
        strategy.max_retries = 2 if profile.quality_score < 50 else 1

        # ---- Log the complete strategy decision ----
        self.logger.log_decision(
            agent="DecisionEngine",
            action=f"Selected strategy: engine={strategy.ocr_engine}, preprocessing={strategy.preprocessing}",
            reasoning="; ".join(strategy.reasoning),
            alternatives=["easyocr", "tesseract", "paddleocr", "trocr"],
            confidence=min(1.0, profile.quality_score / 100.0),
            metadata={
                "image_quality": profile.quality_score,
                "brightness": profile.brightness,
                "blur_score": profile.blur_score,
                "density": profile.density,
            },
        )

        return strategy

    def _select_engine(self, profile: ImageProfile, strategy: ProcessingStrategy) -> str:
        """Select the best OCR engine based on image characteristics and history."""

        # Check if user has a preferred engine
        preferred = self.memory.get_preference("preferred_engine")
        if preferred:
            strategy.reasoning.append(f"User prefers '{preferred}' engine")
            return preferred

        # Check historical performance
        best_from_history = self.memory.get_best_engine()

        # Rule-based selection with image characteristics
        engine = "tesseract"  # default (easyocr requires local deployment with GPU/high RAM)

        if profile.density == "dense":
            engine = "tesseract"
            strategy.reasoning.append("Dense text detected → Tesseract selected (better for dense pages)")

        elif profile.is_blurry:
            engine = "tesseract"
            strategy.reasoning.append("Blurry image detected → Tesseract selected with aggressive preprocessing")

        elif profile.is_dark or profile.is_low_contrast:
            engine = "tesseract"
            strategy.reasoning.append("Poor image quality → Tesseract with enhanced preprocessing")

        elif profile.quality_score > 80 and best_from_history:
            # Good quality + historical data → use what worked best
            engine = best_from_history
            strategy.reasoning.append(
                f"High quality image + historical data → {best_from_history} "
                f"(best avg confidence from past runs)"
            )

        else:
            strategy.reasoning.append("Standard conditions → Tesseract selected")

        # Safety check
        check = self.safety.validate_action("select_ocr_engine", {"engine": engine})
        if not check.is_safe:
            engine = "tesseract"
            strategy.reasoning.append(f"Safety override: reverted to tesseract ({check.blocked_reason})")

        return engine

    def _decide_preprocessing(self, profile: ImageProfile, strategy: ProcessingStrategy) -> None:
        """Decide preprocessing steps based on image analysis."""

        if profile.is_low_resolution:
            strategy.apply_upscale = True
            strategy.reasoning.append("Low resolution → upscaling enabled")

        if profile.is_dark:
            strategy.apply_brightness_enhance = True
            strategy.reasoning.append(f"Dark image (brightness={profile.brightness:.0f}) → brightness enhancement enabled")

        if profile.is_low_contrast:
            strategy.apply_contrast_enhance = True
            strategy.reasoning.append(f"Low contrast (std={profile.contrast:.0f}) → CLAHE enhancement enabled")

        if profile.is_blurry:
            strategy.apply_sharpening = True
            strategy.reasoning.append(f"Blurry image (score={profile.blur_score:.0f}) → sharpening enabled")

        if profile.is_skewed:
            strategy.apply_deskew = True
            strategy.reasoning.append(f"Skewed by {profile.skew_angle:.1f}° → deskew enabled")

        # Determine overall preprocessing level
        enhancements = sum([
            strategy.apply_upscale,
            strategy.apply_brightness_enhance,
            strategy.apply_contrast_enhance,
            strategy.apply_sharpening,
            strategy.apply_deskew,
        ])

        if enhancements == 0:
            strategy.preprocessing = "minimal"
            strategy.reasoning.append("Good image quality → minimal preprocessing")
        elif enhancements <= 2:
            strategy.preprocessing = "standard"
        else:
            strategy.preprocessing = "aggressive"
            strategy.reasoning.append("Multiple issues detected → aggressive preprocessing")

    def _decide_dual_channel(self, profile: ImageProfile, strategy: ProcessingStrategy) -> None:
        """Decide whether to use dual-channel (red ink) extraction."""
        if profile.dominant_color == "red_ink":
            check = self.safety.validate_action("enable_dual_channel")
            if check.is_safe:
                strategy.use_dual_channel = True
                strategy.reasoning.append("Red ink detected → dual-channel OCR enabled")

    def suggest_retry_strategy(
        self,
        current_strategy: ProcessingStrategy,
        quality_score: float,
        attempt: int,
    ) -> Optional[ProcessingStrategy]:
        """
        If OCR quality is below threshold, suggest a different strategy for retry.
        Returns None if no retry should be attempted.
        """
        if attempt >= current_strategy.max_retries:
            return None

        if quality_score >= current_strategy.confidence_threshold:
            return None

        retry = ProcessingStrategy()
        retry.reasoning = [f"Retry #{attempt + 1}: quality {quality_score:.2f} below threshold {current_strategy.confidence_threshold:.2f}"]

        # Retry with different PSM modes / aggressive preprocessing — stay on tesseract
        retry.ocr_engine = "tesseract"
        retry.reasoning.append("Retry with Tesseract + aggressive preprocessing")

        # Escalate preprocessing
        retry.preprocessing = "aggressive"
        retry.apply_upscale = True
        retry.apply_contrast_enhance = True
        retry.apply_sharpening = True
        retry.apply_deskew = True
        retry.confidence_threshold = current_strategy.confidence_threshold
        retry.max_retries = current_strategy.max_retries

        self.logger.log_decision(
            agent="DecisionEngine",
            action=f"Retry strategy: switching to {retry.ocr_engine} with aggressive preprocessing",
            reasoning="; ".join(retry.reasoning),
            alternatives=[current_strategy.ocr_engine],
            confidence=0.5,
        )

        return retry
