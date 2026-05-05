"""
Orchestrator — Master Agent Controller
========================================
The Orchestrator is the top-level coordinator that implements the full
Observe → Interpret → Decide → Act → Learn cycle (Slide 25).

It wires together all agent components:
  - PerceptionAgent (Observe)
  - DecisionEngine (Decide)
  - Phase 1 Core (Act — OCR, formatting, docx generation)
  - FeedbackLoop (Learn)
  - MemoryStore (Remember)
  - AgentLogger (Log)
  - SafetyGuard (Validate)
  - PrivacyGuard (Protect)

Phase 2 Requirement Mapping:
  - Slide 25 (Operational Workflow: Observe → Interpret → Decide → Act → Learn)
  - Slide 23 (Agent Architecture: complete flow)
  - Slide 20 (Agentic System Concept)
  - Slide 22 (Agentic Vision: Tool → Agent)
"""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from src.agents.perception_agent import PerceptionAgent, ImageProfile
from src.agents.decision_engine import DecisionEngine, ProcessingStrategy
from src.agents.memory_store import MemoryStore, ConversionRecord
from src.agents.feedback_loop import FeedbackLoop, QualityReport
from src.agents.agent_logger import AgentLogger
from src.agents.safety_guard import SafetyGuard
from src.agents.privacy_guard import PrivacyGuard, PrivacyReport

from src.ocr_engine import OCREngine
from src.formatting_detector import FormattingDetector, FormattedBlock
from src.docx_generator import DocxGenerator
from src.preprocessing import (
    preprocess,
    preprocess_for_easyocr,
    preprocess_with_red_extraction,
)


@dataclass
class AgentResult:
    """Complete result of an agentic conversion."""
    text: str = ""
    blocks: list = field(default_factory=list)
    image_profile: Optional[ImageProfile] = None
    strategy: Optional[ProcessingStrategy] = None
    quality_report: Optional[QualityReport] = None
    privacy_report: Optional[PrivacyReport] = None
    corrections_applied: int = 0
    retry_count: int = 0
    api_ocr_used: bool = False
    success: bool = False
    error: str = ""


class Orchestrator:
    """
    Master agent that coordinates the full agentic pipeline.

    Phase 1 Pipeline (static):
        open image → preprocess → OCR → format → save

    Phase 2 Pipeline (agentic):
        OBSERVE (analyze image) →
        INTERPRET (perception profile) →
        DECIDE (strategy selection) →
        ACT (adaptive OCR + formatting) →
        EVALUATE (quality check) →
        RETRY? (if quality is poor) →
        PROTECT (privacy scan) →
        LEARN (store results, apply corrections) →
        OUTPUT

    The key difference: Phase 1 follows the SAME path every time.
    Phase 2 ADAPTS its path based on the image and past experience.
    """

    def __init__(
        self,
        memory: Optional[MemoryStore] = None,
        logger: Optional[AgentLogger] = None,
        safety: Optional[SafetyGuard] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> None:
        # Initialize all agent components
        self.memory = memory or MemoryStore()
        self.logger = logger or AgentLogger()
        self.safety = safety or SafetyGuard(autonomy_level="semi")

        self.perception = PerceptionAgent()
        self.decision = DecisionEngine(self.memory, self.logger, self.safety)
        self.feedback = FeedbackLoop(self.memory, self.logger)
        self.privacy = PrivacyGuard()

        self.formatter = FormattingDetector()
        self.doc_generator = DocxGenerator()

        # UI callbacks
        self._on_status = on_status or (lambda s: None)
        self._on_progress = on_progress or (lambda p: None)

        self.logger.log_decision(
            agent="Orchestrator",
            action="Agentic system initialized",
            reasoning="All agent modules loaded and ready",
            confidence=1.0,
        )

    def process_image(self, image_path: Path | str) -> AgentResult:
        """
        Full agentic pipeline: Observe → Decide → Act → Evaluate → Learn.

        This replaces the static Phase 1 _ocr_task() method.
        """
        result = AgentResult()
        path = Path(image_path)

        try:
            # ============================================================ #
            #  STEP 1: OBSERVE — Perception Agent analyzes the image        #
            # ============================================================ #
            self._on_status("🔍 Step 1/6: Analyzing image...")
            self._on_progress(10)

            profile = self.perception.analyze(path)
            result.image_profile = profile

            self.logger.log_decision(
                agent="PerceptionAgent",
                action=f"Image analyzed: quality={profile.quality_score}/100",
                reasoning=f"brightness={profile.brightness:.0f}, contrast={profile.contrast:.0f}, "
                          f"blur={profile.blur_score:.0f}, density={profile.density}",
                confidence=1.0,
                metadata={"recommendations": profile.recommendations},
            )

            # ============================================================ #
            #  STEP 2: DECIDE — Decision Engine selects strategy             #
            # ============================================================ #
            self._on_status("🧠 Step 2/6: Deciding optimal strategy...")
            self._on_progress(20)

            strategy = self.decision.decide_strategy(profile)
            result.strategy = strategy

            # ============================================================ #
            #  STEP 3: ACT — Execute OCR with the chosen strategy            #
            # ============================================================ #
            self._on_status(f"⚡ Step 3/6: Running {strategy.ocr_engine} OCR...")
            self._on_progress(40)

            text, boxes, confidence_values = self._execute_ocr(path, strategy)

            # ============================================================ #
            #  STEP 4: EVALUATE — Feedback Loop assesses quality             #
            # ============================================================ #
            self._on_status("📊 Step 4/6: Evaluating quality...")
            self._on_progress(60)

            quality = self.feedback.evaluate_quality(
                text, confidence_values, strategy.confidence_threshold
            )
            result.quality_report = quality

            # ---- Retry if quality is unacceptable ----
            attempt = 0
            while not quality.is_acceptable and attempt < strategy.max_retries:
                attempt += 1
                retry_strategy = self.decision.suggest_retry_strategy(
                    strategy, quality.overall_score, attempt
                )
                if retry_strategy is None:
                    break

                self._on_status(f"🔄 Retry {attempt}: Trying {retry_strategy.ocr_engine}...")

                retry_text, retry_boxes, retry_conf = self._execute_ocr(path, retry_strategy)
                retry_quality = self.feedback.evaluate_quality(
                    retry_text, retry_conf, retry_strategy.confidence_threshold
                )

                if retry_quality.overall_score > quality.overall_score:
                    text = retry_text
                    boxes = retry_boxes
                    confidence_values = retry_conf
                    quality = retry_quality
                    strategy = retry_strategy
                    result.retry_count = attempt

                    self.logger.log_decision(
                        agent="Orchestrator",
                        action=f"Retry {attempt} improved quality: {quality.overall_score:.2f}",
                        reasoning="Retry produced better results, adopting new output",
                        confidence=quality.overall_score,
                    )

            result.quality_report = quality
            result.strategy = strategy

            # ============================================================ #
            #  STEP 4b: ESCALATE — Vision API when local OCR quality fails  #
            # ============================================================ #
            # The agent autonomously decides to call an external Vision API
            # only when local engines have exhausted their retries and the
            # output is still clearly unusable (score < 0.35). This is a
            # real cost/privacy tradeoff decision — logged for transparency.
            if not quality.is_acceptable and quality.overall_score < 0.35:
                api_result = self._try_api_escalation(path, quality)
                if api_result is not None:
                    api_text, api_quality = api_result
                    if api_quality.overall_score > quality.overall_score:
                        text = api_text
                        boxes = []
                        quality = api_quality
                        result.quality_report = quality
                        result.api_ocr_used = True
                        result.retry_count += 1

            # ============================================================ #
            #  STEP 5: LEARN — Apply corrections & update memory             #
            # ============================================================ #
            self._on_status("📝 Step 5/6: Applying learned corrections...")
            self._on_progress(75)

            # Apply previously learned corrections
            text, corrections_applied = self.feedback.apply_learned_corrections(text)
            result.corrections_applied = corrections_applied

            # ============================================================ #
            #  STEP 6: PROTECT — Privacy scan                               #
            # ============================================================ #
            self._on_status("🔒 Step 6/6: Scanning for privacy concerns...")
            self._on_progress(85)

            privacy_report = self.privacy.scan_text(text)
            result.privacy_report = privacy_report

            if privacy_report.has_pii:
                self.logger.log_decision(
                    agent="PrivacyGuard",
                    action=f"PII detected: {len(privacy_report.detections)} item(s), risk={privacy_report.risk_level}",
                    reasoning="Sensitive data found in OCR output",
                    confidence=1.0,
                    metadata={"warnings": privacy_report.warnings},
                )

            # ============================================================ #
            #  FINALIZE — Build formatted blocks and store in memory         #
            # ============================================================ #
            self._on_status("✅ Finalizing...")
            self._on_progress(95)

            if boxes:
                result.blocks = self.formatter.detect_formatting(boxes)
            else:
                result.blocks = [
                    FormattedBlock(
                        text=text,
                        block_type="body",
                        alignment="left",
                        indent_level=0,
                    )
                ]

            result.text = text
            result.success = True

            # Store conversion in memory
            record = ConversionRecord(
                image_path=str(path),
                engine_used=strategy.ocr_engine,
                preprocessing_strategy=strategy.preprocessing,
                quality_score=quality.overall_score,
                confidence=quality.avg_confidence,
                image_brightness=profile.brightness,
                image_blur_score=profile.blur_score,
                success=True,
            )
            self.memory.add_conversion(record)

            self._on_progress(100)
            self._on_status("✅ Done — Agent pipeline complete")

        except Exception as e:
            result.success = False
            result.error = str(e)
            self._on_status(f"❌ Error: {e}")

            self.logger.log_decision(
                agent="Orchestrator",
                action="Pipeline failed",
                reasoning=str(e),
                confidence=0.0,
                outcome="error",
            )

        return result

    def save_document(
        self,
        blocks: list,
        output_path: Path | str,
        original_text: str = "",
        edited_text: str = "",
    ) -> bool:
        """
        Save the document — with safety check and learning.

        If the user edited the text before saving, the Feedback Loop
        learns from the corrections.
        """
        # Safety check for save action
        has_pii = False
        if edited_text:
            pii_report = self.privacy.scan_text(edited_text)
            has_pii = pii_report.has_pii

        check = self.safety.validate_action("save_document", {"has_pii": has_pii})

        # Learn from user corrections
        if original_text and edited_text and original_text != edited_text:
            corrections = self.feedback.learn_from_corrections(original_text, edited_text)

            # Update the last conversion record
            history = self.memory.get_history(limit=1)
            if history:
                last = history[0]
                last.was_corrected = True
                last.user_corrections = corrections

        # Generate the document
        success = self.doc_generator.generate_docx(blocks, output_path)

        self.logger.log_decision(
            agent="Orchestrator",
            action=f"Document saved to {output_path}",
            reasoning="User triggered save",
            confidence=1.0,
            outcome="success" if success else "failed",
        )

        return success

    def get_decision_log(self, limit: int = 20) -> str:
        """Return the human-readable decision log."""
        return self.logger.get_display_log(limit)

    def get_engine_stats(self) -> Dict[str, Any]:
        """Return engine performance statistics from memory."""
        return self.memory.get_engine_stats()

    # ================================================================ #
    #  Private: OCR execution with adaptive preprocessing                #
    # ================================================================ #

    def _execute_ocr(
        self,
        image_path: Path,
        strategy: ProcessingStrategy,
    ) -> Tuple[str, list, List[float]]:
        """
        Execute OCR using the strategy determined by the Decision Engine.

        This replaces the static Phase 1 approach of always using the
        same preprocessing + engine combination.
        """
        # ---- Adaptive preprocessing ----
        img = self._apply_adaptive_preprocessing(image_path, strategy)

        # ---- Run the selected OCR engine ----
        engine = OCREngine(engine=strategy.ocr_engine)
        text = engine.run(img)

        # Try to get boxes for formatting detection
        boxes = []
        confidence_values = []
        try:
            boxes = engine.run_with_boxes(img)
            confidence_values = [
                float(b.get("confidence", 0.0)) for b in boxes
            ]
        except (NotImplementedError, Exception):
            pass

        # ---- Dual-channel if enabled ----
        if strategy.use_dual_channel:
            try:
                _, red_img = preprocess_with_red_extraction(image_path)
                red_text = engine.run(red_img)
                if red_text.strip():
                    text = self._merge_dual_channel(text, red_text)

                    self.logger.log_decision(
                        agent="Orchestrator",
                        action="Dual-channel merge completed",
                        reasoning="Red ink text merged with main text",
                        confidence=0.8,
                    )
            except Exception:
                pass  # non-critical — fall back to single-channel

        return text, boxes, confidence_values

    def _apply_adaptive_preprocessing(
        self,
        image_path: Path,
        strategy: ProcessingStrategy,
    ) -> np.ndarray:
        """
        Apply preprocessing steps based on the strategy.
        
        Unlike Phase 1 (one-size-fits-all), this adapts based on the
        image profile and the Decision Engine's choices.
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        if strategy.preprocessing == "minimal":
            # Minimal: just return the raw image (good quality images)
            self.logger.log_decision(
                agent="Orchestrator",
                action="Minimal preprocessing applied",
                reasoning="Image quality is good — minimal processing preserves detail",
                confidence=0.9,
            )
            return img

        # ---- Apply selected enhancements ----

        if strategy.apply_upscale:
            h, w = img.shape[:2]
            if w < 2000:
                scale = 2000 / float(w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if strategy.apply_brightness_enhance or strategy.apply_contrast_enhance:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if strategy.apply_contrast_enhance:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

            if strategy.apply_brightness_enhance:
                # Adaptive brightness correction
                mean_brightness = float(np.mean(gray))
                if mean_brightness < 90:
                    gamma = 90.0 / max(1.0, mean_brightness)
                    gamma = min(gamma, 2.5)
                    table = np.array([
                        ((i / 255.0) ** (1.0 / gamma)) * 255
                        for i in range(256)
                    ]).astype("uint8")
                    gray = cv2.LUT(gray, table)

            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if strategy.apply_sharpening:
            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
            img = cv2.filter2D(img, -1, kernel)
            img = np.clip(img, 0, 255).astype(np.uint8)

        if strategy.apply_deskew:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 12,
            )
            coords = cv2.findNonZero(binary)
            if coords is not None and len(coords) > 100:
                rect = cv2.minAreaRect(coords)
                angle = rect[-1]
                # Normalize OpenCV 4.5+ angle convention to signed near 0
                if angle > 45:
                    angle = angle - 90
                elif angle < -45:
                    angle = angle + 90
                # Skip rotation for implausible angles (likely misdetection)
                if 0.2 < abs(angle) <= 15:
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)
                    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    img = cv2.warpAffine(
                        img, matrix, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )

        return img

    def _merge_dual_channel(self, main_text: str, red_text: str) -> str:
        """Merge main and red-ink channel text, deduplicating."""
        merged_lines = []
        seen = set()
        for line in (main_text + "\n" + red_text).splitlines():
            clean = line.strip()
            key = clean.lower()
            if not clean or key in seen:
                continue
            seen.add(key)
            merged_lines.append(clean)
        return "\n".join(merged_lines)

    # ================================================================ #
    #  API escalation — Vision API fallback when local OCR fails        #
    # ================================================================ #

    def _try_api_escalation(
        self,
        image_path: Path,
        current_quality: QualityReport,
    ) -> Optional[Tuple[str, QualityReport]]:
        """
        Attempt OCR via the OpenAI Vision API as a last resort.

        The agent reaches this path only when:
          1. All local engines have been tried and retried
          2. Quality score is still below 0.35 (clearly unusable output)

        This is an autonomous agent decision with a cost/privacy tradeoff:
          - Cost: an external API call is made (not free)
          - Privacy: the image leaves the local machine
          - Benefit: Vision models handle handwriting far better than
            local OCR engines

        Blocked in "manual" autonomy mode — the user must act explicitly.
        Allowed in "semi" and "full" modes with a logged privacy notice.
        """
        if self.safety.autonomy_level == "manual":
            self.logger.log_decision(
                agent="DecisionEngine",
                action="API escalation skipped — manual autonomy mode",
                reasoning=(
                    "autonomy_level=manual: agent will not send image data to "
                    "an external service without an explicit user action"
                ),
                confidence=1.0,
            )
            return None

        api_key = self._read_api_key()
        if not api_key:
            self.logger.log_decision(
                agent="DecisionEngine",
                action="API escalation skipped — no OPENAI_API_KEY in .env",
                reasoning="Key not found; set OPENAI_API_KEY in .env to enable Vision API fallback",
                confidence=1.0,
            )
            return None

        self.logger.log_decision(
            agent="DecisionEngine",
            action="Escalating to Vision API (gpt-4o-mini)",
            reasoning=(
                f"Local OCR quality={current_quality.overall_score:.2f} after all retries — "
                f"issues: {'; '.join(current_quality.issues) or 'fragmented output'}. "
                f"Vision API selected as final fallback."
            ),
            alternatives=["accept poor quality", "request manual correction"],
            confidence=0.9,
            metadata={
                "autonomy_level": self.safety.autonomy_level,
                "privacy_note": "Image will be sent to OpenAI API",
            },
        )
        self.logger.log_decision(
            agent="PrivacyGuard",
            action="API escalation privacy notice",
            reasoning=(
                "Image is being sent to OpenAI gpt-4o-mini for OCR. "
                "Review the image for sensitive/confidential content. "
                "This is logged for transparency and audit."
            ),
            confidence=1.0,
        )

        try:
            self._on_status("🌐 Escalating to Vision API (gpt-4o-mini)...")
            api_text = self._run_api_ocr(image_path, api_key)

            if not api_text.strip():
                self.logger.log_decision(
                    agent="Orchestrator",
                    action="API OCR returned empty result",
                    reasoning="Vision API produced no text — keeping local OCR output",
                    confidence=0.0,
                    outcome="rejected",
                )
                return None

            api_quality = self.feedback.evaluate_quality(api_text, threshold=0.5)

            self.logger.log_decision(
                agent="FeedbackLoop",
                action=f"API OCR quality: {api_quality.overall_score:.2f} "
                       f"({'ACCEPTED' if api_quality.overall_score > current_quality.overall_score else 'REJECTED'})",
                reasoning=(
                    f"Vision API result — words={api_quality.word_count}, "
                    f"confidence={api_quality.avg_confidence:.2f}. "
                    f"Previous local score was {current_quality.overall_score:.2f}."
                ),
                confidence=api_quality.overall_score,
                outcome=(
                    "api_result_adopted"
                    if api_quality.overall_score > current_quality.overall_score
                    else "api_result_rejected_no_improvement"
                ),
            )

            return api_text, api_quality

        except Exception as exc:
            self.logger.log_decision(
                agent="Orchestrator",
                action="API escalation failed",
                reasoning=str(exc),
                confidence=0.0,
                outcome="error",
            )
            return None

    def _run_api_ocr(self, image_path: Path, api_key: str) -> str:
        """Call the OpenAI Vision API and return extracted text."""
        with open(image_path, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("ascii")
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an OCR assistant. Extract text exactly as it appears.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all visible text from this image. "
                                "Preserve line breaks and structure. "
                                "Return only the extracted text — no explanations."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "temperature": 0,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _read_api_key(self) -> Optional[str]:
        """Read OPENAI_API_KEY from project .env or environment."""
        # src/agents/orchestrator.py → parents[2] = project root
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "OPENAI_API_KEY":
                    return value.strip().strip('"').strip("'")
        return os.getenv("OPENAI_API_KEY")
