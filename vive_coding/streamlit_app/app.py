import streamlit as st
from streamlit.components.v1 import html

st.set_page_config(
    page_title="Flutter Web App",
    layout="wide"
)

# Flutter 앱을 iframe으로 로드
html_code = """
<iframe src="http://localhost:8501/static/web/index.html" 
        width="100%" 
        height="800"
        frameborder="0"
        style="border:none;">
</iframe>
"""

html(html_code) 