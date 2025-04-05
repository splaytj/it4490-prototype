import pdfplumber
from transformers import pipeline
import difflib

gec_pipeline = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")

# Extract text from submitted pdf
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Corrects grammar using GEC model.
def correct_grammar(text):
    corrected = gec_pipeline(text, max_length=512)[0]['generated_text']
    return corrected

# Extracts text, corrects grammar, compares text side-by-side
def process_resume(pdf_file):
    original_text = extract_text_from_pdf(pdf_file)
    corrected_text = correct_grammar(original_text)
    
    # Generate HTML diff to highlight changes
    differ = difflib.HtmlDiff()
    diff_html = differ.make_file(original_text.splitlines(), corrected_text.splitlines(), "Original", "Corrected")
    return diff_html
