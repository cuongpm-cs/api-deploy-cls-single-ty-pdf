import fitz
from pdf2image import convert_from_bytes
import pytesseract
from io import BytesIO

TESSERACT_PATH = r'Tesseract-OCR/tesseract.exe'
POPPLER_PATH = r'poppler-24.07.0/Library/bin'

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text_from_pdf_normal(file_stream):
    file_stream.seek(0)
    with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
        return " ".join(page.get_text().replace("\n", " ") for page in doc).strip()

def extract_text_with_ocr(file_stream):
    file_stream.seek(0)
    images = convert_from_bytes(file_stream.read(), poppler_path=POPPLER_PATH)
    text = ""
    for image in images:
        ocr_text = pytesseract.image_to_string(image, lang='vie')
        text += ocr_text.replace('\n', ' ').strip() + " "
    return text.strip()

import requests

def pdf_converter(pdf_path_or_url, min_bytes=512):
    try:
        if pdf_path_or_url.startswith("http://") or pdf_path_or_url.startswith("https://"):
            response = requests.get(pdf_path_or_url)
            response.raise_for_status()
            file_bytes = response.content
        else:
            with open(pdf_path_or_url, 'rb') as f:
                file_bytes = f.read()

        text = extract_text_from_pdf_normal(BytesIO(file_bytes))
        if len(text.encode("utf-8")) < min_bytes:
            text = extract_text_with_ocr(BytesIO(file_bytes))
        return text.strip()
    except Exception as e:
        print(f"❌ Lỗi khi xử lý PDF: {e}")
        return ""
