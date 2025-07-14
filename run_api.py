# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import re, os
# import numpy as np
# import joblib
# from pdf_converter import pdf_converter

# app = Flask(__name__)
# CORS(app)

# mlb = joblib.load('mlb.pkl')

# tokenizer_ty_cqbh = AutoTokenizer.from_pretrained(
#     'models-ty-cqbh', use_fast=False
# )
# model_ty_cqbh = AutoModelForSequenceClassification.from_pretrained(
#     'models-ty-cqbh',
#     num_labels=3,
#     problem_type="multi_label_classification"
# )

# tokenizer_full = AutoTokenizer.from_pretrained(
#     'models-full-v2', use_fast=False
# )
# model_full = AutoModelForSequenceClassification.from_pretrained(
#     'models-full-v2',
#     num_labels=3,
#     problem_type="multi_label_classification"
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cuda")

# model_ty_cqbh.to(device)
# model_full.to(device)

# model_ty_cqbh.eval()
# model_full.eval()

# def clean_content(text):
#     text = str(text).lower()
#     text = re.sub(r"[^a-zà-ỹ0-9\s]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# def clean_content_pdf(text):
#     text = str(text).lower()
    
#     text = re.sub(r'ký bởi[:：]?\s?.*?(?=[\n\r\.]|$)', '', text)
    
#     patterns = [
#         r"cộng hòa xã hội chủ nghĩa việt nam",
#         r"độc lập\s*-\s*tự do\s*-\s*hạnh phúc",
#     ]
#     for pat in patterns:
#         text = re.sub(pat, '', text, flags=re.IGNORECASE | re.MULTILINE)

#     text = re.sub(r"[^a-zà-ỹ0-9\s]", " ", text)

#     text = re.sub(r"\s+", " ", text).strip()

#     return text

# def classification_ty_cqbh(trich_yeu, co_quan_ban_hanh):
#     raw_text = trich_yeu + " do cơ quan " + co_quan_ban_hanh + " ban hành"
#     cleaned_text = clean_content(raw_text)

#     inputs = tokenizer_ty_cqbh(
#         cleaned_text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=128
#     )
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = model_ty_cqbh(**inputs)
#         logits = outputs.logits
#         probs = torch.sigmoid(logits).squeeze().cpu().numpy()

#     pred_binary = (probs > 0.5).astype(int)
#     pred_binary = np.array([pred_binary]) 
#     predicted_labels = mlb.inverse_transform(pred_binary)[0]
#     return predicted_labels

# def classification_full(trich_yeu, co_quan_ban_hanh, noi_dung):
#     ty_cqbh = trich_yeu + " do cơ quan " + co_quan_ban_hanh + " ban hành"
#     ty_cqbh_cleaned = clean_content(ty_cqbh)
#     nd_cleaned = clean_content_pdf(noi_dung)
#     text_input = ty_cqbh_cleaned + "Nội dung văn bản: " + nd_cleaned

#     inputs = tokenizer_full(
#         text_input,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=256
#     )
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = model_full(**inputs)
#         logits = outputs.logits
#         probs = torch.sigmoid(logits).squeeze().cpu().numpy()

#     pred_binary = (probs > 0.5).astype(int)
#     pred_binary = np.array([pred_binary]) 
#     predicted_labels = mlb.inverse_transform(pred_binary)[0]
#     return predicted_labels

# @app.route('/cls_ty_cqbh', methods=['POST'])
# def receive_data_ty_cqbh():
#     trich_yeu = request.json.get('trich_yeu')
#     co_quan_ban_hanh = request.json.get('co_quan_ban_hanh')

#     if not trich_yeu or not co_quan_ban_hanh:
#         return jsonify({'error': 'Missing required fields'}), 400

#     output = classification_ty_cqbh(trich_yeu, co_quan_ban_hanh)
#     return jsonify({'output': output})

# @app.route('/cls_full', methods=['POST'])
# def receive_data_full():
#     trich_yeu = request.json.get('trich_yeu')
#     co_quan_ban_hanh = request.json.get('co_quan_ban_hanh')
#     pdf_path = request.json.get('pdf_path')

#     if not trich_yeu or not co_quan_ban_hanh:
#         return jsonify({'error': 'Missing required fields'}), 400

#     output = classification_full(trich_yeu, co_quan_ban_hanh, pdf_converter(pdf_path))
#     return jsonify({'output': output})

# if __name__ == "__main__":
#     app.run(debug=True, port=8888)

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# import re, os, numpy as np, joblib
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from pdf_converter import pdf_converter

# app = Flask(__name__)
# CORS(app)

# # Load multi-label binarizer
# mlb = joblib.load('mlb.pkl')

# # Load models & tokenizers
# def load_model(model_path, num_labels):
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_path,
#         num_labels=num_labels,
#         problem_type="multi_label_classification"
#     ).to(device)
#     model.eval()
#     return tokenizer, model

# device = torch.device("cuda" if torch.cuda.is_available() else "cuda")

# tokenizer_ty_cqbh, model_ty_cqbh = load_model('models-ty-cqbh', num_labels=3)
# tokenizer_full, model_full = load_model('models-full-v2', num_labels=3)

# # ==============================
# #       TEXT CLEANING
# # ==============================
# def clean_content(text):
#     text = str(text).lower()
#     text = re.sub(r"[^a-zà-ỹ0-9\s]", " ", text)
#     return re.sub(r"\s+", " ", text).strip()

# def clean_content_pdf(text):
#     text = str(text).lower()
#     text = re.sub(r'ký bởi[:：]?\s?.*?(?=[\n\r\.]|$)', '', text)
#     patterns = [
#         r"cộng hòa xã hội chủ nghĩa việt nam",
#         r"độc lập\s*-\s*tự do\s*-\s*hạnh phúc",
#     ]
#     for pat in patterns:
#         text = re.sub(pat, '', text, flags=re.IGNORECASE)
#     text = re.sub(r"[^a-zà-ỹ0-9\s]", " ", text)
#     return re.sub(r"\s+", " ", text).strip()

# # ==============================
# #        INFERENCE LOGIC
# # ==============================
# def classify(model, tokenizer, text, max_length=256):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True,
#                        padding=True, max_length=max_length)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         logits = model(**inputs).logits
#         probs = torch.sigmoid(logits).squeeze().cpu().numpy()

#     pred_binary = np.array([(probs > 0.5).astype(int)])
#     return mlb.inverse_transform(pred_binary)[0]

# # ==============================
# #          ROUTES
# # ==============================
# @app.route('/cls_ty_cqbh', methods=['POST'])
# def classify_ty_cqbh():
#     data = request.json
#     trich_yeu = data.get('trich_yeu')
#     co_quan = data.get('co_quan_ban_hanh')

#     if not trich_yeu or not co_quan:
#         return jsonify({'error': 'Missing required fields'}), 400

#     raw_text = f"{trich_yeu} do cơ quan {co_quan} ban hành"
#     cleaned = clean_content(raw_text)
#     labels = classify(model_ty_cqbh, tokenizer_ty_cqbh, cleaned, max_length=128)
#     return jsonify({'output': labels})

# @app.route('/cls_full', methods=['POST'])
# def classify_full():
#     data = request.json
#     trich_yeu = data.get('trich_yeu')
#     co_quan = data.get('co_quan_ban_hanh')
#     pdf_file = data.get('pdf_file')

#     if not trich_yeu or not co_quan or not pdf_file:
#         return jsonify({'error': 'Missing required fields'}), 400

#     header = clean_content(f"{trich_yeu} do cơ quan {co_quan} ban hành")
#     content = clean_content_pdf(pdf_converter(pdf_file))
#     combined_text = f"{header}Nội dung văn bản: {content}"

#     labels = classify(model_full, tokenizer_full, combined_text, max_length=256)
#     return jsonify({'output': labels})

# if __name__ == "__main__":
#     app.run(debug=True, port=8888)

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import re, numpy as np, joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pdf_converter import pdf_converter  # Đảm bảo module này tương thích

app = FastAPI()

# Cho phép CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
#     MODEL LOADING
# =========================
# mlb = joblib.load('mlb.pkl')

def load_model(model_path, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        problem_type="single_label_classification"
    ).to(device)
    model.eval()
    return tokenizer, model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer_ty_cqbh, model_ty_cqbh = load_model('models-ty-cqbh', num_labels=3)
tokenizer_ty_pdf, model_ty_pdf = load_model('models-cuc-cds-single-phobert-base-finetuned-full-last-v3-32/snapshots/e143971cf58ba84cff079c347888a0b97194332d', num_labels=3)

# =========================
#     TEXT CLEANING
# =========================
def clean_content(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zà-ỹ0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def clean_content_pdf(text):
    text = str(text).lower()
    text = re.sub(r'ký bởi[:：]?\s?.*?(?=[\n\r\.]|$)', '', text)
    patterns = [
        r"cộng hòa xã hội chủ nghĩa việt nam",
        r"độc lập\s*-\s*tự do\s*-\s*hạnh phúc",
    ]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zà-ỹ0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# =========================
#     INFERENCE LOGIC
# =========================
index2label = {
    0: 'Cục Biến Đổi Khí Hậu',
    1: 'Cục Công Nghệ Thông Tin',
    2: 'Cục Đo Đạc Bản Đồ'
}

def classify(model, tokenizer, text, max_length=256):
    inputs = tokenizer(text, 
                       return_tensors="pt", 
                       truncation=True,
                       padding='max_length', 
                       max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    pred_label_idx = int(np.argmax(probs))
    return index2label[pred_label_idx]

# =========================
#     REQUEST MODELS
# =========================
class TY_PDF_Request(BaseModel):
    trich_yeu: str
    pdf_file: str  # Đảm bảo đầu vào này là dạng base64 hoặc path nếu local

# =========================
#         ROUTES
# =========================
# @app.post("/cls_ty_cqbh")
# async def classify_ty_cqbh(payload: TY_CQBH_Request):
#     if not payload.trich_yeu or not payload.co_quan_ban_hanh:
#         raise HTTPException(status_code=400, detail="Missing required fields")

#     raw_text = f"{payload.trich_yeu} do cơ quan {payload.co_quan_ban_hanh} ban hành"
#     cleaned = clean_content(raw_text)
#     labels = classify(model_ty_cqbh, tokenizer_ty_cqbh, cleaned, max_length=128)

#     return {"output": labels}


@app.post("/cls_ty_pdf")
async def classify_full(payload: TY_PDF_Request):
    if not payload.trich_yeu or not payload.pdf_file:
        raise HTTPException(status_code=400, detail="Missing required fields")

    trich_yeu = clean_content(payload.trich_yeu)
    content = clean_content_pdf(pdf_converter(payload.pdf_file))
    print(content)
    combined_text = f"{trich_yeu} nội dung văn bản: {content}"

    labels = classify(model_ty_pdf, tokenizer_ty_pdf, combined_text, max_length=256)
    return {"output": labels}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run_api:app", host="0.0.0.0", port=8888, reload=False)