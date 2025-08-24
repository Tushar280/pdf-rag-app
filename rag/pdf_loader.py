import pdfplumber

def extract_text_by_page(path):
    with pdfplumber.open(path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]

def store_pdf(path):
    # Placeholder function for storing PDFs, can be expanded as needed.
    return True
