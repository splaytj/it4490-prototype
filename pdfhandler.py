import pdfplumber
from transformers import pipeline
import difflib

# Load the grammar correction model
gec_pipeline = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def correct_grammar(text):
    """Correct grammar using the GEC model."""
    # For simplicity, assume text fits within the model's 512-token limit
    corrected = gec_pipeline(text, max_length=512)[0]['generated_text']
    return corrected

def process_resume(pdf_file):
    """Process the resume: extract text, correct grammar, and show differences."""
    original_text = extract_text_from_pdf(pdf_file)
    corrected_text = correct_grammar(original_text)
    
    # Generate HTML diff to highlight changes
    differ = difflib.HtmlDiff()
    diff_html = differ.make_file(original_text.splitlines(), corrected_text.splitlines(), "Original", "Corrected")
    return diff_html