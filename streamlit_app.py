import streamlit as st
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Must be the very first Streamlit call
st.set_page_config(
    page_title="Image-to-Word Agentic System",
    page_icon="🤖",
    layout="wide"
)




import sys
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.agents.orchestrator import Orchestrator, AgentResult
from src.agents.memory_store import MemoryStore
from src.agents.agent_logger import AgentLogger
from src.agents.safety_guard import SafetyGuard
from src.formatting_detector import FormattedBlock

# ── Writable temp dir for memory/log files (safe on all platforms incl. Streamlit Cloud) ──
_TMP_DIR = Path(tempfile.gettempdir()) / "agentic_ocr_data"
_TMP_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Session State
if "memory" not in st.session_state:
    st.session_state.memory = MemoryStore(memory_path=_TMP_DIR / "agent_memory.json")
if "logger" not in st.session_state:
    st.session_state.logger = AgentLogger(log_path=_TMP_DIR / "agent_decisions.jsonl")
if "safety" not in st.session_state:
    st.session_state.safety = SafetyGuard(autonomy_level="semi")
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator(
        memory=st.session_state.memory,
        logger=st.session_state.logger,
        safety=st.session_state.safety,
        on_status=lambda s: st.session_state.update({"status_msg": s}),
        on_progress=lambda p: st.session_state.update({"progress_val": p}),
    )

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "original_ocr_text" not in st.session_state:
    st.session_state.original_ocr_text = ""
if "status_msg" not in st.session_state:
    st.session_state.status_msg = "Ready — Upload an image to begin"
if "progress_val" not in st.session_state:
    st.session_state.progress_val = 0

st.title("Image-to-Word Converter — Agentic System (Phase 2)")

# Sidebar Controls
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    
    autonomy_level = st.selectbox(
        "Autonomy Level", 
        ["full", "semi", "manual"], 
        index=1
    )
    if autonomy_level != st.session_state.safety.get_autonomy_level():
        st.session_state.safety.set_autonomy_level(autonomy_level)
        st.session_state.logger.log_decision(
            agent="User",
            action=f"Autonomy level changed to '{autonomy_level}'",
            reasoning="User preference",
            confidence=1.0,
        )

    run_agent_btn = st.button("🤖 Run Agent", disabled=not uploaded_file, use_container_width=True)
    
    st.divider()
    
    if st.button("🗑 Clear Memory", use_container_width=True):
        if st.session_state.memory.memory_path.exists():
            os.remove(st.session_state.memory.memory_path)
        st.session_state.memory = MemoryStore(memory_path=_TMP_DIR / "agent_memory.json")
        st.session_state.orchestrator.memory = st.session_state.memory
        st.session_state.orchestrator.decision.memory = st.session_state.memory
        st.session_state.orchestrator.feedback.memory = st.session_state.memory
        st.success("Memory cleared!")

# Layout
col1, col2, col3 = st.columns([1, 1.5, 1])

# Left Column: Image Previews and Agent Info
with col1:
    st.subheader("Image & Profile")
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Original Image", width='stretch')
        
    with st.expander("📊 Image Profile", expanded=False):
        if st.session_state.last_result and st.session_state.last_result.image_profile:
            profile = st.session_state.last_result.image_profile
            st.markdown(f"""
            **Resolution:** {profile.width} × {profile.height}  
            **Brightness:** {profile.brightness:.1f} {'⚠ DARK' if profile.is_dark else '✓'}  
            **Contrast:** {profile.contrast:.1f} {'⚠ LOW' if profile.is_low_contrast else '✓'}  
            **Blur Score:** {profile.blur_score:.1f} {'⚠ BLURRY' if profile.is_blurry else '✓'}  
            **Skew Angle:** {profile.skew_angle:.1f}° {'⚠ SKEWED' if profile.is_skewed else '✓'}  
            **Text Density:** {profile.density}  
            **Color Profile:** {profile.dominant_color}  
            
            **★ Quality Score:** {profile.quality_score}/100
            
            **─── Recommendations ───**
            """)
            for rec in profile.recommendations:
                st.markdown(f"- {rec}")
        else:
            st.text("Run agent to see image analysis...")

# --- Execution Logic ---
if run_agent_btn and uploaded_file is not None:
    if "docx_bytes" in st.session_state:
        del st.session_state.docx_bytes

    # We must run the orchestrator and update progress
    with st.spinner("🤖 Agent pipeline running..."):
        # Save uploaded file to temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Disable EasyOCR (not available on Streamlit Cloud — too heavy).
        # The OCREngine auto-mode will fall back to Tesseract gracefully.
        from src.ocr_engine import OCREngine
        OCREngine._cached_easyocr_reader = None

        progress_bar = st.progress(0, text="Initializing...")
        
        # Override callbacks temporarily for this run to update UI natively
        def _update_status(s):
            st.session_state.status_msg = s
            progress_bar.progress(st.session_state.progress_val, text=s)
            
        def _update_progress(p):
            st.session_state.progress_val = p
            progress_bar.progress(p, text=st.session_state.status_msg)
            
        st.session_state.orchestrator._on_status = _update_status
        st.session_state.orchestrator._on_progress = _update_progress
        
        try:
            result = st.session_state.orchestrator.process_image(tmp_path)
            st.session_state.last_result = result
            st.session_state.original_ocr_text = result.text
            
            if result.success:
                st.success("✅ Agent pipeline complete!")
            else:
                st.error(f"❌ Agent failed: {result.error}")
        finally:
            os.remove(tmp_path)
            # Need to rerun to propagate results to other columns
            st.rerun()

# Main Column: Extracted Text
with col2:
    st.subheader("Extracted Text")
    
    current_text = ""
    if st.session_state.last_result:
        current_text = st.session_state.last_result.text
        
    edited_text = st.text_area("Review and edit extracted text", value=current_text, height=400)
    
    # Save Action
    if st.session_state.last_result and st.session_state.last_result.success:
        if st.button("💾 Generate .docx", type="primary", use_container_width=True):
            has_pii = False
            pii_report = st.session_state.orchestrator.privacy.scan_text(edited_text)
            has_pii = pii_report.has_pii
            
            check = st.session_state.safety.validate_action("save_document", {"has_pii": has_pii})
            if check.is_safe or (check.requires_human_approval and st.session_state.safety.get_autonomy_level() != "manual"):
                blocks = [
                    FormattedBlock(
                        text=edited_text,
                        block_type="body",
                        alignment="left",
                        indent_level=0,
                    )
                ]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    temp_docx_path = tmp.name
                    
                success = st.session_state.orchestrator.save_document(
                    blocks=blocks,
                    output_path=temp_docx_path,
                    original_text=st.session_state.original_ocr_text,
                    edited_text=edited_text,
                )
                
                if success:
                    with open(temp_docx_path, "rb") as f:
                        docx_bytes = f.read()
                    st.session_state.docx_bytes = docx_bytes
                else:
                    st.error("Failed to generate document.")
            else:
                st.warning(f"Action blocked by Safety Guard: {check.warnings}")

        if "docx_bytes" in st.session_state:
            st.success("Document successfully generated!")
            st.download_button(
                label="⬇️ Download .docx file",
                data=st.session_state.docx_bytes,
                file_name="extracted_document.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )

# Right Column: Agent Log, Privacy & Quality
with col3:
    st.subheader("Agent Output")
    
    tab1, tab2, tab3 = st.tabs(["🧠 Log", "🔒 Privacy", "📊 Quality"])
    
    with tab1:
        st.text_area("Agent Decision Log", st.session_state.logger.get_display_log(30), height=300, disabled=True, label_visibility="collapsed")
        
    with tab2:
        if st.session_state.last_result and st.session_state.last_result.privacy_report:
            report = st.session_state.last_result.privacy_report
            if not report.has_pii:
                st.success("✅ No PII detected. Document appears safe.")
            else:
                st.error(f"⚠ PRIVACY ALERT — Risk Level: {report.risk_level.upper()}")
                st.markdown("**─── Detections ───**")
                for det in report.detections:
                    st.markdown(f"- {det.pii_type}: {det.value} (risk: {det.risk_level})")
                st.markdown("**─── Warnings ───**")
                for w in report.warnings:
                    st.markdown(f"- {w}")
                st.markdown("**─── Recommendations ───**")
                for r in report.recommendations:
                    st.markdown(f"- {r}")
        else:
            st.text("Privacy analysis will appear here...")
            
    with tab3:
        if st.session_state.last_result and st.session_state.last_result.quality_report:
            report = st.session_state.last_result.quality_report
            status = "✅ PASS" if report.is_acceptable else "❌ FAIL"
            st.markdown(f"""
            **Overall Score:** {report.overall_score:.2f} {status}  
            **Avg Confidence:** {report.avg_confidence:.2f}  
            **Words:** {report.word_count} | **Lines:** {report.line_count}  
            **Gibberish Ratio:** {report.gibberish_ratio:.1%}  
            **Short Word Ratio:** {report.short_word_ratio:.1%}
            """)
            if report.issues:
                st.markdown("**─── Issues ───**")
                for issue in report.issues:
                    st.markdown(f"- ⚠ {issue}")
            st.markdown(f"💡 {report.suggestion}")
            if st.session_state.last_result.retry_count > 0:
                st.markdown(f"🔄 Retries used: {st.session_state.last_result.retry_count}")
            if st.session_state.last_result.corrections_applied > 0:
                st.markdown(f"📝 Learned corrections applied: {st.session_state.last_result.corrections_applied}")
        else:
            st.text("Quality report will appear here...")
