import os
import fitz
from docx import Document
import pandas as pd
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup

def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext in [".txt", ".md"]:
        return open(file_path, "r", encoding="utf-8").read()
    elif ext == ".docx":
        return "\n".join([p.text for p in Document(file_path).paragraphs])
    elif ext in [".csv", ".xlsx"]:
        df = pd.read_excel(file_path) if ext == ".xlsx" else pd.read_csv(file_path)
        return "\n".join(df.astype(str).fillna("").to_string(index=False).split("\n"))
    elif ext in [".jpg", ".jpeg", ".png"]:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)
    elif ext in [".html", ".htm"]:
        html = open(file_path, "r", encoding="utf-8").read()
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def extract_pdf(file_path):
    text = []
    pdf = fitz.open(file_path)
    for page in pdf:
        page_text = page.get_text()
        if not page_text.strip():  # OCR if empty
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img)
        text.append(page_text)
    return "\n".join(text)

def chunk_text(text, size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i+size]))
    return chunks
