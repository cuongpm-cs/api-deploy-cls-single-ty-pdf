from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import re, numpy as np, joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pdf_converter import pdf_converter 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

tokenizer_ty_pdf, model_ty_pdf = load_model('models-cuc-cds-single-phobert-base-finetuned-full-last-v3-32/snapshots/e143971cf58ba84cff079c347888a0b97194332d', num_labels=3)

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

class TY_PDF_Request(BaseModel):
    trich_yeu: str
    pdf_file: str  

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
