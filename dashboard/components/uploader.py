"""
File uploader component for Streamlit dashboard.
"""
import streamlit as st
from typing import Optional, Tuple
import io


def file_uploader() -> Optional[Tuple[bytes, str, str]]:
    """
    Display file uploader and return file data.
    Returns: (file_bytes, file_name, file_type) or None
    """
    st.subheader("📄 Upload Medical Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF or image file",
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
        help="Upload a medical prescription, report, or diagnosis document"
    )
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        file_type = uploaded_file.type
        
        st.success(f"✅ File uploaded: {file_name}")
        st.info(f"File type: {file_type}, Size: {len(file_bytes)} bytes")
        
        return file_bytes, file_name, file_type
    
    return None

