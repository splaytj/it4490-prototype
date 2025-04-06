import pdfplumber
from transformers import pipeline
import difflib
import re
 
gec_pipeline = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
 
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text
 
def correct_grammar(text):
    corrected = gec_pipeline(text, max_length=512)[0]['generated_text']
    return corrected
 
def process_resume(pdf_file):
    original_text = extract_text_from_pdf(pdf_file)
    corrected_text = correct_grammar(original_text)
    # Generate HTML diff
    differ = difflib.HtmlDiff(wrapcolumn=80)
    diff_html = differ.make_file(
        original_text.splitlines(), 
        corrected_text.splitlines(), 
        "Original", 
        "Corrected"
    )
 
    # 1. Remove the default style block difflib inserts
    diff_html = re.sub(r'(?s)<style type="text/css">.*?</style>', '', diff_html)
    # 2. Remove any inline style="..." attributes to prevent forced white text
    diff_html = re.sub(r'style="[^"]*"', '', diff_html)
    # 3. Inject a custom CSS style block with !important overrides for text color
    custom_style = """
<style type="text/css">
  /* Force white background and black text on everything */
  html, body, table.diff, table.diff td, table.diff th {
    background-color: white !important;
    color: black !important;
  }
  /* Keep the diff color highlights but ensure text is black */
  .diff_header {
    background-color: #f0f0f0 !important;
    color: black !important;
    padding: 5px;
    text-align: center;
    font-weight: bold;
  }
  .diff_next {
    background-color: #e0e0e0 !important;
    color: black !important;
  }
  .diff_add {
    background-color: #d0ffd0 !important; /* light green */
    color: black !important;
  }
  .diff_chg {
    background-color: #ffffcc !important; /* light yellow */
    color: black !important;
  }
  .diff_sub {
    background-color: #ffd0d0 !important; /* light red */
    color: black !important;
  }
  table.diff {
    border: 1px solid #ccc;
    width: 100%;
    border-collapse: collapse;
  }
</style>
"""
    # Prepend the custom style to the HTML output
    diff_html = custom_style + diff_html
    return diff_html