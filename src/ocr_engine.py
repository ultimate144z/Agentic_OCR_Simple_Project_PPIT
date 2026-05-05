from __future__ import annotations

import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


CHEMISTRY_TERMS = [
    "adsorption",
    "absorption",
    "desorption",
    "surface",
    "chemistry",
    "enthalpy",
    "entropy",
    "gibbs",
    "energy",
    "physical",
    "chemical",
    "coagulation",
    "colloidal",
    "catalysis",
    "electrophoresis",
    "turbidity",
    "micelles",
    "lyophilic",
    "lyophobic",
    "emulsion",
    "aerosol",
    "sol",
    "gel",
    "reaction",
    "temperature",
    "pressure",
    "isotherm",
    "freundlich",
    "langmuir",
    "electrolyte",
]

CHEMISTRY_CORRECTIONS = {
    "adsoaphon": "adsorption",
    "desoephon": "desorption",
    "scefoce": "surface",
    "molecuan": "molecular",
    "foeunolhch": "freundlich",
    "lonqmuit": "langmuir",
    "exothedmic": "exothermic",
    "physiosooption": "physisorption",
    "chemisodf": "chemisorption",
    "enexc": "energy",
    "sohol": "solid",
    "hqui": "liquid",
    "balk": "bulk",
    "coloidoi": "colloidal",
    "soluliog": "solution",
    "acsoxpton": "adsorption",
    "ocsosphon": "adsorption",
}


class OCREngine:
    def __init__(self, engine: str = "auto") -> None:
        normalized = engine.lower().strip()
        supported = {"auto", "tesseract", "easyocr", "paddleocr", "trocr"}
        if normalized not in supported:
            raise ValueError(f"Unsupported OCR engine: {engine}")

        self.engine = normalized
        self._easyocr_reader = None
        self.last_confidence_rows: List[Dict[str, Any]] = []
        self._tesseract_psm_modes = [6, 3, 4, 11]

        self._configure_tesseract()

    def run(self, image: Path | str | np.ndarray) -> str:
        if self.engine == "auto":
            # Only attempt EasyOCR if a pre-warmed reader is available.
            # If not (e.g. Streamlit Cloud deployment), skip straight to Tesseract.
            if getattr(OCREngine, "_cached_easyocr_reader", None) is not None:
                try:
                    text = self._run_easyocr_text(image)
                except Exception:
                    text = ""
                if text.strip():
                    return self._postprocess_text(text)
            return self._postprocess_text(self._run_tesseract_text(image))

        if self.engine == "easyocr":
            return self._postprocess_text(self._run_easyocr_text(image))

        if self.engine == "tesseract":
            return self._postprocess_text(self._run_tesseract_text(image))

        if self.engine == "paddleocr":
            return self._postprocess_text(self._run_paddleocr_text(image))

        if self.engine == "trocr":
            return self._postprocess_text(self._run_trocr_text(image))

        return ""

    def run_with_boxes(self, image: Path | str | np.ndarray) -> List[Dict[str, Any]]:
        if self.engine == "auto":
            if getattr(OCREngine, "_cached_easyocr_reader", None) is not None:
                try:
                    boxes = self._run_easyocr_boxes(image)
                except Exception:
                    boxes = []
                if boxes:
                    self.last_confidence_rows = self._to_confidence_rows(boxes)
                    return boxes
            boxes = self._run_tesseract_boxes(image)
            self.last_confidence_rows = self._to_confidence_rows(boxes)
            return boxes

        if self.engine == "easyocr":
            boxes = self._run_easyocr_boxes(image)
            self.last_confidence_rows = self._to_confidence_rows(boxes)
            return boxes

        if self.engine == "tesseract":
            boxes = self._run_tesseract_boxes(image)
            self.last_confidence_rows = self._to_confidence_rows(boxes)
            return boxes

        if self.engine in {"paddleocr", "trocr"}:
            raise NotImplementedError(f"{self.engine} box extraction is currently disabled in this module")

        return []

    def run_both_channels(
        self,
        main_img: Path | str | np.ndarray,
        red_img: Path | str | np.ndarray,
    ) -> str:
        main_text = self.run(main_img)
        red_text = self.run(red_img)

        merged_lines: List[str] = []
        seen = set()
        for line in (main_text + "\n" + red_text).splitlines():
            clean = line.strip()
            key = clean.lower()
            if not clean or key in seen:
                continue
            seen.add(key)
            merged_lines.append(clean)

        return "\n".join(merged_lines)

    def _configure_tesseract(self) -> None:
        found = shutil.which("tesseract")
        if not found:
            default_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            if default_path.exists():
                found = str(default_path)
        if found:
            pytesseract.pytesseract.tesseract_cmd = found

    def _load_image(self, image: Path | str | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image

        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return img

    def _run_tesseract_text(self, image: Path | str | np.ndarray) -> str:
        img = self._load_image(image)
        best_text = ""
        best_score = -1.0

        for psm in self._tesseract_psm_modes:
            config = self._tesseract_config(psm)
            text = pytesseract.image_to_string(img, config=config)
            score = self._score_text(text)
            if score > best_score:
                best_score = score
                best_text = text

        return best_text

    def _run_tesseract_boxes(self, image: Path | str | np.ndarray) -> List[Dict[str, Any]]:
        img = self._load_image(image)
        best_boxes: List[Dict[str, Any]] = []
        best_score = -1.0

        for psm in self._tesseract_psm_modes:
            config = self._tesseract_config(psm)
            data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
            boxes = self._parse_tesseract_data(data)
            score = self._score_boxes(boxes)
            if score > best_score:
                best_score = score
                best_boxes = boxes

        return best_boxes

    def _parse_tesseract_data(self, data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        boxes: List[Dict[str, Any]] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = str(data["text"][i]).strip()
            if not text:
                continue

            raw_conf = data.get("conf", ["-1"])[i]
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue

            boxes.append(
                {
                    "text": text,
                    "x": int(data["left"][i]),
                    "y": int(data["top"][i]),
                    "w": int(data["width"][i]),
                    "h": int(data["height"][i]),
                    "confidence": conf / 100.0,
                }
            )

        boxes.sort(key=lambda b: (b["y"], b["x"]))
        return boxes

    def _run_paddleocr_text(self, image: Path | str | np.ndarray) -> str:
        from paddleocr import PaddleOCR

        img = self._load_image(image)
        ocr = PaddleOCR(lang="en")
        result = ocr.ocr(img)
        if not result:
            return ""

        lines: List[str] = []
        if isinstance(result, list):
            first_item = result[0] if result else None
            if isinstance(first_item, list):
                for line in first_item:
                    if isinstance(line, (list, tuple)) and len(line) > 1:
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and text_info:
                            lines.append(str(text_info[0]))
            elif isinstance(first_item, dict):
                rec_texts = first_item.get("rec_texts")
                if isinstance(rec_texts, list):
                    lines = [str(t) for t in rec_texts]

        return "\n".join(lines)

    def _run_trocr_text(self, image: Path | str | np.ndarray) -> str:
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        # TrOCR performs better on line/word crops than full-page notes.
        # We tile 384x384 patches to reduce the single-patch truncation issue,
        # but it remains less reliable than EasyOCR for these handwritten pages.
        img = self._load_image(image)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        patch_texts: List[str] = []
        for patch in self._tile_image_for_trocr(pil_img, tile_size=384):
            pixel_values = processor(images=patch, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if decoded:
                patch_texts.append(decoded)

        return "\n".join(patch_texts)

    def _tile_image_for_trocr(self, image, tile_size: int = 384):
        width, height = image.size
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                crop = image.crop((x, y, min(x + tile_size, width), min(y + tile_size, height)))
                yield crop

    def _tesseract_config(self, psm: int) -> str:
        return f"--oem 1 --psm {psm}"

    def _score_text(self, text: str) -> float:
        if not text:
            return 0.0

        alpha_num_chars = sum(ch.isalnum() for ch in text)
        total_chars = max(1, len(text))
        density = alpha_num_chars / total_chars

        words = [w for w in re.findall(r"[A-Za-z]{2,}|\d+", text)]
        word_bonus = min(200, len(words)) / 200.0

        return density + word_bonus

    def _score_boxes(self, boxes: List[Dict[str, Any]]) -> float:
        if not boxes:
            return 0.0

        avg_conf = sum(float(b.get("confidence", 0.0)) for b in boxes) / len(boxes)
        count_bonus = min(300, len(boxes)) / 300.0
        return avg_conf + count_bonus

    def _get_easyocr_reader(self):
        # Use pre-warmed reader injected by Streamlit cache_resource if available
        if hasattr(OCREngine, "_cached_easyocr_reader") and OCREngine._cached_easyocr_reader is not None:
            return OCREngine._cached_easyocr_reader
        if self._easyocr_reader is None:
            import easyocr
            self._easyocr_reader = easyocr.Reader(["en"], gpu=False)
        return self._easyocr_reader

    def _run_easyocr_text(self, image: Path | str | np.ndarray) -> str:
        reader = self._get_easyocr_reader()
        img = self._load_image(image)
        results = reader.readtext(img, detail=1, paragraph=False)
        ordered = self._order_easyocr_results(results)
        return "\n".join(ordered)

    def _run_easyocr_boxes(self, image: Path | str | np.ndarray) -> List[Dict[str, Any]]:
        reader = self._get_easyocr_reader()
        img = self._load_image(image)
        results = reader.readtext(img, detail=1, paragraph=False)

        boxes: List[Dict[str, Any]] = []
        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue

            box, text, conf = item[0], str(item[1]).strip(), float(item[2])
            if not text or conf < 0.15:
                continue
            if not (isinstance(box, (list, tuple)) and len(box) >= 4):
                continue

            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0 = int(min(xs))
            y0 = int(min(ys))
            x1 = int(max(xs))
            y1 = int(max(ys))

            boxes.append(
                {
                    "text": text,
                    "x": x0,
                    "y": y0,
                    "w": max(0, x1 - x0),
                    "h": max(0, y1 - y0),
                    "confidence": conf,
                }
            )

        boxes.sort(key=lambda b: (b["y"], b["x"]))
        return boxes

    def _order_easyocr_results(self, results: list) -> List[str]:
        extracted: List[tuple[float, float, str]] = []

        for idx, item in enumerate(results):
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue

            box, text, conf = item[0], str(item[1]).strip(), float(item[2])
            if not text or conf < 0.15:
                continue

            if isinstance(box, (list, tuple)) and box and isinstance(box[0], (list, tuple)):
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                min_x = float(min(xs))
                min_y = float(min(ys))
            else:
                min_x = 0.0
                min_y = float(idx * 20)

            extracted.append((min_y, min_x, text))

        if not extracted:
            return []

        x_values = [x for _, x, _ in extracted]
        min_x = min(x_values)
        max_x = max(x_values)
        split_x = (min_x + max_x) / 2.0

        has_two_columns = False
        if len(extracted) >= 16 and (max_x - min_x) > 350:
            left_count = sum(1 for _, x, _ in extracted if x <= split_x)
            right_count = len(extracted) - left_count
            has_two_columns = left_count >= 4 and right_count >= 4

        if has_two_columns:
            extracted.sort(key=lambda row: (0 if row[1] <= split_x else 1, row[0], row[1]))
        else:
            extracted.sort(key=lambda row: (row[0], row[1]))

        lines: List[str] = []
        current_line = [extracted[0][2]]
        current_y = extracted[0][0]
        current_col = 0 if (has_two_columns and extracted[0][1] <= split_x) else 1

        for y, x, text in extracted[1:]:
            col = 0 if (has_two_columns and x <= split_x) else 1
            if col == current_col and abs(y - current_y) <= 18:
                current_line.append(text)
            else:
                lines.append(" ".join(current_line))
                current_line = [text]
                current_y = y
                current_col = col
        lines.append(" ".join(current_line))

        return [line for line in lines if line.strip()]

    def _postprocess_text(self, text: str) -> str:
        text = text.replace("|", " ").replace("~", " ").replace("_", " ").replace("@", "a")

        lines: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            tokens = re.findall(r"[A-Za-z]+|\d+|[^A-Za-z\d\s]", line)
            repaired = [self._best_term_match(token) for token in tokens]
            rebuilt = " ".join(repaired)
            rebuilt = re.sub(r"\s+([,.;:!?])", r"\1", rebuilt)
            rebuilt = re.sub(r"\s+", " ", rebuilt).strip()
            if rebuilt:
                lines.append(rebuilt)

        cleaned_text = "\n".join(lines)
        return self._reconstruct_subscripts(cleaned_text)

    def _best_term_match(self, token: str) -> str:
        # Misread corrections
        token = token.replace("rn", "m")
        token = token.replace("cl", "d")
        token = token.replace("li", "h")

        lower = token.lower()
        if lower in CHEMISTRY_CORRECTIONS:
            corrected = CHEMISTRY_CORRECTIONS[lower]
            if token[:1].isupper():
                return corrected.capitalize()
            return corrected

        if len(lower) < 4 or not lower.isalpha():
            return token

        best = lower
        best_score = 0.0
        for term in CHEMISTRY_TERMS:
            score = SequenceMatcher(None, lower, term).ratio()
            if score > best_score:
                best_score = score
                best = term

        if best_score >= 0.78:
            # Character fixes based on context (alpha word)
            best = best.replace("0", "o").replace("1", "l")
            if token[0].isupper():
                return best.capitalize()
            return best

        # Contextual character fixes if not a recognized term but looks like alpha word
        if len(token) >= 2 and any(c.isalpha() for c in token):
            token = token.replace("0", "O") if token.isupper() else token.replace("0", "o")
            token = token.replace("1", "l")

        return token

    def _reconstruct_subscripts(self, text: str) -> str:
        subscript_map = {
            "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
            "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉"
        }
        
        def replace_match(match):
            prefix = match.group(1)
            digits = match.group(2)
            subscripts = "".join(subscript_map.get(d, d) for d in digits)
            return prefix + subscripts

        # Match Chemical Element + Digits (e.g., H2, O2, SO4, C12)
        return re.sub(r"([A-Z][a-z]?)(\d+)", replace_match, text)

    def _to_confidence_rows(self, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for box in boxes:
            rows.append(
                {
                    "text": box.get("text", ""),
                    "confidence": float(box.get("confidence", 0.0)),
                    "x": int(box.get("x", 0)),
                    "y": int(box.get("y", 0)),
                }
            )
        return rows
