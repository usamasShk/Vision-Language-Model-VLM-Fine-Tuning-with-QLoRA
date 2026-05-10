"""
Document to Markdown Converter
A Streamlit app that converts document images to Markdown using a fine-tuned Vision Language Model.
"""

import streamlit as st
import torch
import gc
import sys
import logging
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
import io
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="📄 Doc to Markdown",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 30px;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #1f77b4;
        }
        .success-box {
            background-color: #d4edda;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #28a745;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
INSTRUCTION = "Convert this document image to Markdown format. Preserve all formatting including headings, bold, italics, lists, tables, and equations. Output only valid Markdown."
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
WEIGHTS_PATH = Path(__file__).parent / "Weights"

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    st.header("ℹ️ About This App")
    st.info("""
    **Document to Markdown Converter**
    
    This app uses a fine-tuned Vision Language Model (Qwen2-VL-2B-Instruct) 
    with QLoRA to convert document images into structured Markdown format.
    
    **How it works:**
    1. Upload an image of your document
    2. Model loads on first use (2-3 minutes)
    3. Click Convert to generate Markdown
    4. Download the result
    
    **Features:**
    - Preserves formatting (headings, lists, tables)
    - Handles equations and special characters
    - Optimized for Streamlit Cloud
    """)

# ============================================================================
# MODEL LOADING (CACHED WITH ERROR HANDLING)
# ============================================================================
@st.cache_resource
def load_model_and_processor():
    """Load the model and processor with robust error handling."""
    try:
        logger.info("Starting model and processor loading...")
        gc.collect()
        
        device = "cpu"  # Force CPU for Streamlit Cloud compatibility
        logger.info(f"Using device: {device}")
        
        # Load processor with timeout
        logger.info(f"Loading processor...")
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            timeout=120
        )
        logger.info("✓ Processor loaded")
        
        # Load base model with optimizations
        logger.info(f"Loading base model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            timeout=300
        )
        logger.info("✓ Base model loaded")
        
        # Load LoRA adapter if available
        if WEIGHTS_PATH.exists():
            logger.info(f"Loading LoRA adapter...")
            model = PeftModel.from_pretrained(model, WEIGHTS_PATH)
            model.eval()
            logger.info("✓ LoRA adapter loaded")
        
        logger.info("✓ All models ready")
        return model, processor, device
    
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        return None, None, None

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================
def convert_image_to_markdown(image: Image.Image, model, processor, device) -> str:
    """Convert an image to Markdown using the fine-tuned model."""
    try:
        logger.info("Starting inference...")
        
        # Prepare the image
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize image if too large
        max_size = 1024
        if image.size[0] > max_size or image.size[1] > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            logger.info(f"Image resized to {image.size}")
        
        # Prepare messages in ChatML format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": INSTRUCTION}
                ]
            }
        ]
        
        # Process text
        logger.info("Processing inputs...")
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        logger.info(f"Inputs moved to {device}")
        
        # Generate markdown
        logger.info("Generating markdown...")
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        logger.info("Decoding output...")
        # Decode output
        markdown_text = processor.decode(
            output_ids[0],
            skip_special_tokens=True
        ).split("assistant\n")[-1].strip()
        
        logger.info("✓ Inference complete")
        return markdown_text
    
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise Exception(f"Inference failed: {str(e)}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
st.markdown("""
<div class="main-header">
    <h1>📄 Document to Markdown Converter</h1>
    <p>Convert document images to structured Markdown using AI</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for model (lazy loading)
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.processor = None
    st.session_state.device = None

# File uploader
st.subheader("📤 Upload Document")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg", "bmp"],
    help="Upload a scanned document or document image"
)

# Process uploaded file and load model on demand (lazy loading)
if uploaded_file is not None:
    # Load model only once when file is first uploaded
    if not st.session_state.model_loaded:
        with st.spinner("⏳ Loading AI model (this may take 2-3 minutes on first run)..."):
            try:
                logger.info("Lazy loading: Attempting to load model after file upload...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                model, processor, device = load_model_and_processor()
                
                if model is not None and processor is not None:
                    st.session_state.model = model
                    st.session_state.processor = processor
                    st.session_state.device = device
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("❌ Failed to load model. Please try again.")
                    st.stop()
            except Exception as e:
                logger.error(f"Lazy load failed: {str(e)}", exc_info=True)
                st.error(f"❌ Error loading model: {str(e)}")
                st.info("💡 Try refreshing the page or uploading a different image.")
                st.stop()
    
    # Access loaded model from session state
    model = st.session_state.model
    processor = st.session_state.processor
    device = st.session_state.device
    # File size check
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 50:
        st.error(f"❌ File size ({file_size_mb:.2f}MB) exceeds limit (50MB)")
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Document")
                st.image(image, use_column_width=True)
                
                # Display image info
                st.markdown(f"""
                <div class="info-box">
                <b>Image Info:</b><br>
                Size: {image.size[0]} × {image.size[1]} px<br>
                Format: {image.format}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("✨ Converted Markdown")
                
                # Convert button
                if st.button("🚀 Convert to Markdown", use_container_width=True):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("🔄 Converting image to Markdown...")
                        progress_bar.progress(30)
                        
                        start_time = time.time()
                        markdown_text = convert_image_to_markdown(image, model, processor, device)
                        conversion_time = time.time() - start_time
                        
                        progress_bar.progress(100)
                        status_text.text(f"✅ Conversion complete in {conversion_time:.2f}s")
                        
                        # Display markdown
                        st.markdown("""---""")
                        st.markdown(markdown_text)
                        
                        # Download buttons
                        st.markdown("""---""")
                        col_copy, col_download = st.columns(2)
                        
                        with col_copy:
                            st.text_area(
                                "Markdown Output",
                                value=markdown_text,
                                height=200,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                        
                        with col_download:
                            st.download_button(
                                label="📥 Download as .md",
                                data=markdown_text,
                                file_name=f"{uploaded_file.name.split('.')[0]}.md",
                                mime="text/markdown",
                                use_container_width=True
                            )
                            
                            # Also offer txt download
                            st.download_button(
                                label="📥 Download as .txt",
                                data=markdown_text,
                                file_name=f"{uploaded_file.name.split('.')[0]}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        progress_bar.empty()
                        status_text.empty()
                    
                    except Exception as e:
                        logger.error(f"Conversion error: {str(e)}", exc_info=True)
                        st.error(f"❌ Conversion failed: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()
        
        except Exception as e:
            logger.error(f"Image loading error: {str(e)}", exc_info=True)
            st.error(f"❌ Error loading image: {str(e)}")
else:
    st.info("👆 Upload an image file to get started. The model will load on first use.")

# Footer
st.markdown("""---""")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>Document to Markdown Converter | Fine-tuned with QLoRA on Qwen2-VL-2B</p>
    <p>Dataset: Nougat Training Dataset | 2 Epochs | Batch Size: 4</p>
</div>
""", unsafe_allow_html=True)
