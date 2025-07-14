# import fitz
# from pdf2image import convert_from_path
# import pytesseract
# from PIL import Image

# def extract_text_from_pdf_normal(pdf_path):
#     with fitz.open(pdf_path) as doc:
#         return " ".join(page.get_text().replace("\n", " ") for page in doc)

# def pdf_scan_to_txt(pdf_path):
#     pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR/tesseract.exe'
#     images = convert_from_path(
#         pdf_path, 
#         poppler_path=r'poppler-24.07.0/Library/bin'
#     )
#     text = ""
#     for i, image in enumerate(images):
#         ocr_result = pytesseract.image_to_string(image, lang='vie')
#         ocr_result = ocr_result.replace('\n', ' ').strip()
#         text += ocr_result + " "
#     return text.strip()

# def pdf_converter(pdf_path):
#     try:
#         text = extract_text_from_pdf_normal(pdf_path)
#         text_size = len(text.encode("utf-8"))

#         if text_size < 512:
#             text = pdf_scan_to_txt(pdf_path)
#             # print(f"⚠️ Text trích xuất quá nhỏ ({text_size} bytes), bỏ qua: {pdf_path}")
#             # continue

#         # txt_filename = os.path.splitext(filename)[0] + ".txt"
#         # txt_path = os.path.join(output_folder, txt_filename)
#         return text.strip()
#     except Exception as e:
#         print("Lỗi trong quá trình convert file pdf!")

# import fitz  # PyMuPDF
# from pdf2image import convert_from_path
# import pytesseract
# import os

# # Cấu hình đường dẫn
# TESSERACT_PATH = r'Tesseract-OCR/tesseract.exe'
# POPPLER_PATH = r'poppler-24.07.0/Library/bin'

# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# def extract_text_from_pdf_normal(pdf_path):
#     with fitz.open(pdf_path) as doc:
#         return " ".join(page.get_text().replace("\n", " ") for page in doc).strip()

# def extract_text_with_ocr(pdf_path):
#     images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
#     text = ""
#     for image in images:
#         ocr_text = pytesseract.image_to_string(image, lang='vie')
#         text += ocr_text.replace('\n', ' ').strip() + " "
#     return text.strip()

# def pdf_converter(pdf_path, min_bytes=512):
#     """
#     Trích xuất text từ PDF. Nếu nội dung quá ngắn, dùng OCR thay thế.

#     Args:
#         pdf_path (str): Đường dẫn đến file PDF
#         min_bytes (int): Ngưỡng tối thiểu cho nội dung hợp lệ (theo bytes)

#     Returns:
#         str: Văn bản trích xuất
#     """
#     try:
#         text = extract_text_from_pdf_normal(pdf_path)
#         if len(text.encode("utf-8")) < min_bytes:
#             text = extract_text_with_ocr(pdf_path)
#         return text.strip()
#     except Exception as e:
#         print(f"❌ Lỗi khi xử lý {pdf_path}: {e}")
#         return ""

# if __name__=="__main__":
#     pass

# print(pdf_converter('../data merge/0. CV Tra loi CV 3002 ngày 14 11 2023_Lam dong_FN_Signed.pdf'))


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

# def pdf_converter(file_stream, min_bytes=512):
#     """
#     Trích xuất text từ BytesIO của PDF. Nếu text quá ngắn, dùng OCR.

#     Args:
#         file_stream (BytesIO): File PDF (dạng bytes)
#         min_bytes (int): Ngưỡng tối thiểu để chấp nhận text (theo bytes)

#     Returns:
#         str: Văn bản trích xuất từ PDF
#     """
#     try:
#         text = extract_text_from_pdf_normal(BytesIO(file_stream.read()))
#         if len(text.encode("utf-8")) < min_bytes:
#             text = extract_text_with_ocr(BytesIO(file_stream.read()))
#         return text.strip()
#     except Exception as e:
#         print(f"❌ Lỗi khi xử lý PDF từ stream: {e}")
#         return ""

# def pdf_converter(pdf_path, min_bytes=512):
#     try:
#         with open(pdf_path, 'rb') as f:
#             file_bytes = f.read() 
#         text = extract_text_from_pdf_normal(BytesIO(file_bytes))
#         if len(text.encode("utf-8")) < min_bytes:
#             text = extract_text_with_ocr(BytesIO(file_bytes))
#         return text.strip()
#     except Exception as e:
#         print(f"❌ Lỗi khi xử lý PDF từ file: {e}")
#         return ""

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




# pdf_path = 'http://localhost:2893/0.%20CV%20xin%20%C3%BD%20ki%E1%BA%BFn%20tra%20loi%20CV%203002%20ng%C3%A0y%2014%2011%202023%20Lam%20dong_Signed.pdf'

# extracted_text = pdf_converter(pdf_path)
# print(extracted_text)