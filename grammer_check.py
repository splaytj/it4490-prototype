import fitz # PyMuPDF
import  language_tool_python
import re

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

def check_grammer(text):
     """Checks grammar using LanguageTool."""
     tool = language_tool_python.LanguageTool('en-US')
     matches = tool.check(text)
     return matches 
def check_resume_format(text):
      """Basic resume format checks."""
      erros = []
      
      # Check for presence of contact details 
      if not re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text):
        errors.append("Missing phone number.")
      if not re.search(r'[\w\.-]+@[\w\.-]+', text):
        errors.append("Missing email address.")
    
    # Check for section headings
      required_sections = ["Experience", "Education", "Skills"]
      for section in required_sections:
        if section.lower() not in text.lower():
            errors.append(f"Missing '{section}' section.")
    
      return errors

def main(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    grammar_issues = check_grammar(text)
    format_issues = check_resume_format(text)
    
    print("Grammar Issues:")
    for issue in grammar_issues:
        print(f"- {issue.ruleDescription}: {issue.sentence}")
    
    print("\nFormat Issues:")
    for issue in format_issues:
        print(f"- {issue}")

if __name__ == "__main__":
    pdf_path = "resume.pdf"  # Replace with the actual path
    main(pdf_path)
