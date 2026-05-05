from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FormattedBlock:
    text: str
    block_type: str  # "heading", "body", "bullet", "numbered", "equation"
    is_bold: bool = False
    is_italic: bool = False
    alignment: str = "left"  # "left", "center", "right"
    indent_level: int = 0
    line_y: float = 0.0  # vertical position for ordering
    confidence: float = 1.0


class FormattingDetector:
    def __init__(self) -> None:
        self.line_threshold = 15  # pixels to consider boxes on the same line
        self.para_threshold = 30  # pixels to consider a new paragraph

    def detect_formatting(self, ocr_boxes: List[Dict[str, Any]]) -> List[FormattedBlock]:
        if not ocr_boxes:
            return []

        # Detect if we have two columns
        has_two_columns, split_x = self._detect_two_columns(ocr_boxes)
        
        if has_two_columns:
            left_boxes = [b for b in ocr_boxes if (b["x"] + b["w"]/2) <= split_x]
            right_boxes = [b for b in ocr_boxes if (b["x"] + b["w"]/2) > split_x]
            
            left_blocks = self._process_column(left_boxes)
            right_blocks = self._process_column(right_boxes)
            
            return left_blocks + right_blocks
        else:
            return self._process_column(ocr_boxes)

    def _process_column(self, boxes: List[Dict[str, Any]]) -> List[FormattedBlock]:
        if not boxes:
            return []
            
        # 1. Group boxes into lines
        lines = self._group_into_lines(boxes)
        
        # 2. Group lines into blocks (paragraphs)
        blocks = self._group_into_blocks(lines)
        
        # 3. Analyze each block for formatting
        formatted_blocks = []
        for block_lines in blocks:
            formatted_blocks.append(self._analyze_block(block_lines))
            
        return formatted_blocks

    def _detect_two_columns(self, boxes: List[Dict[str, Any]]) -> tuple[bool, float]:
        if len(boxes) < 10:
            return False, 0.0
            
        x_centers = [b["x"] + b["w"]/2 for b in boxes]
        min_x = min(x_centers)
        max_x = max(x_centers)
        range_x = max_x - min_x
        
        if range_x < 300:  # Too narrow for two columns
            return False, 0.0
            
        split_x = (min_x + max_x) / 2
        
        left_count = sum(1 for x in x_centers if x <= split_x)
        right_count = len(x_centers) - left_count
        
        # Heuristic: both columns should have a significant number of boxes
        if left_count > 4 and right_count > 4:
            pass  # two-column layout detected
            return True, split_x
            
        return False, 0.0

    def _group_into_lines(self, boxes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if not boxes:
            return []
            
        # Sort by Y first, then X
        sorted_boxes = sorted(boxes, key=lambda b: (b["y"], b["x"]))
        
        lines: List[List[Dict[str, Any]]] = []
        if not sorted_boxes:
            return lines
            
        current_line = [sorted_boxes[0]]
        current_y = sorted_boxes[0]["y"]
        
        for box in sorted_boxes[1:]:
            if abs(box["y"] - current_y) <= self.line_threshold:
                current_line.append(box)
            else:
                # Sort the finished line by X
                current_line.sort(key=lambda b: b["x"])
                lines.append(current_line)
                current_line = [box]
                current_y = box["y"]
        
        current_line.sort(key=lambda b: b["x"])
        lines.append(current_line)
        return lines

    def _group_into_blocks(self, lines: List[List[Dict[str, Any]]]) -> List[List[List[Dict[str, Any]]]]:
        if not lines:
            return []
            
        blocks: List[List[List[Dict[str, Any]]]] = []
        current_block = [lines[0]]
        
        # Get average line height for better paragraph detection
        line_heights = [max(b["h"] for b in line) for line in lines]
        avg_height = sum(line_heights) / len(line_heights) if line_heights else 20
        
        para_gap = avg_height * 1.5
        
        for i in range(1, len(lines)):
            prev_line = lines[i-1]
            curr_line = lines[i]
            
            prev_y = max(b["y"] + b["h"] for b in prev_line)
            curr_y = min(b["y"] for b in curr_line)
            
            if (curr_y - prev_y) <= para_gap:
                current_block.append(curr_line)
            else:
                blocks.append(current_block)
                current_block = [curr_line]
                
        blocks.append(current_block)
        return blocks

    def _analyze_block(self, block_lines: List[List[Dict[str, Any]]]) -> FormattedBlock:
        # Combine text from all lines in the block
        text_lines = []
        for line in block_lines:
            text_lines.append(" ".join(b["text"] for b in line))
        
        full_text = "\n".join(text_lines)
        
        # Basic type detection
        block_type = "body"
        first_line_text = text_lines[0].strip() if text_lines else ""
        
        # Heading detection
        if self._is_heading(text_lines):
            block_type = "heading"
        # Bullet detection
        elif self._is_bullet(first_line_text):
            block_type = "bullet"
        # Numbered list detection
        elif self._is_numbered(first_line_text):
            block_type = "numbered"
            
        # Alignment detection (basic)
        alignment = self._detect_alignment(block_lines)
        
        # Indentation detection
        indent_level = self._detect_indentation(block_lines)
        
        # Confidence (average of all boxes)
        all_confs = [b["confidence"] for line in block_lines for b in line]
        avg_conf = sum(all_confs) / len(all_confs) if all_confs else 1.0
        
        return FormattedBlock(
            text=full_text,
            block_type=block_type,
            alignment=alignment,
            indent_level=indent_level,
            line_y=float(block_lines[0][0]["y"]) if block_lines and block_lines[0] else 0.0,
            confidence=avg_conf
        )

    def _detect_indentation(self, block_lines: List[List[Dict[str, Any]]]) -> int:
        if not block_lines:
            return 0
        
        # Find the leftmost X in the whole set of blocks (this would ideally be passed in)
        # For now, we'll just check if the first line is indented relative to the others in the block
        x_starts = [line[0]["x"] for line in block_lines if line]
        if not x_starts:
            return 0
            
        min_x = min(x_starts)
        first_x = x_starts[0]
        
        # If the first line starts significantly to the right of the minimum
        if first_x - min_x > 30:
            return 1
        return 0

    def _is_heading(self, text_lines: List[str]) -> bool:
        if not text_lines:
            return False
        
        text = text_lines[0].strip()
        # Rule: ALL CAPS and relatively short
        if text.isupper() and len(text) < 60:
            return True
        # Rule: Starts with #
        if text.startswith("#"):
            return True
        # Rule: Single line and very short compared to body
        if len(text_lines) == 1 and len(text) < 30:
            return True
            
        return False

    def _is_bullet(self, text: str) -> bool:
        # Regex for common bullet symbols
        bullet_pattern = r"^[•\-\*\·\○\▪\+]\s"
        return bool(re.match(bullet_pattern, text))

    def _is_numbered(self, text: str) -> bool:
        # Regex for numbered or lettered lists: 1. or 1) or a. or a)
        numbered_pattern = r"^(\d+|[a-zA-Z])[\.\)]\s"
        return bool(re.match(numbered_pattern, text))

    def _detect_alignment(self, block_lines: List[List[Dict[str, Any]]]) -> str:
        # Basic heuristic for alignment
        # This would ideally need the full image width to be accurate
        return "left"


def detect_formatting(ocr_boxes: List[Dict[str, Any]]) -> List[FormattedBlock]:
    detector = FormattingDetector()
    return detector.detect_formatting(ocr_boxes)
