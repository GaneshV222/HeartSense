import os
import re
import fitz  # PyMuPDF
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
import io

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    
    if ext == '.pdf':
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text()
                
            if len(text.strip()) < 50 and HAS_OCR:
                text = ""
                for page in doc:
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text += pytesseract.image_to_string(img) + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
            
    elif ext in ['.jpg', '.jpeg', '.png'] and HAS_OCR:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(img)
        except Exception as e:
            print(f"Error reading image: {e}")
            
    return text

def parse_clinical_text(text: str) -> dict:
    fields = {}
    
    age_match = re.search(r'(?i)(?:Age|Patient Age)\s*[:\-]?\s*(\d+)', text)
    if age_match:
        val = int(age_match.group(1))
        if 0 < val < 120:
            fields['age'] = {
                "value": val,
                "confidence": 0.95,
                "source": age_match.group(0)
            }
            
    bp_match = re.search(r'(?i)(?:BP|Blood Pressure|Resting BP)\s*[:\-]?\s*(\d{2,3})(?:\s*/\s*\d{2,3})?', text)
    if bp_match:
        val = int(bp_match.group(1))
        if 80 < val < 250:
            fields['trestbps'] = {
                "value": val,
                "confidence": 0.90,
                "source": bp_match.group(0)
            }
            
    chol_match = re.search(r'(?i)(?:Total Cholesterol|Cholesterol|TC|Serum Cholesterol)\s*[:\-]?\s*(\d{2,3})', text)
    if chol_match:
        val = int(chol_match.group(1))
        if 100 < val < 600:
            fields['chol'] = {
                "value": val,
                "confidence": 0.90,
                "source": chol_match.group(0)
            }
            
    thalach_match = re.search(r'(?i)(?:Max(?:imum)? Heart Rate|Heart Rate Max|Thalach)\s*[:\-]?\s*(\d{2,3})', text)
    if thalach_match:
        val = int(thalach_match.group(1))
        if 60 < val < 250:
            fields['thalach'] = {
                "value": val,
                "confidence": 0.90,
                "source": thalach_match.group(0)
            }
            
    fbs_match = re.search(r'(?i)(?:Fasting Blood Sugar|FBS)\s*[:\-]?\s*(\d{2,3})', text)
    if fbs_match:
        val = int(fbs_match.group(1))
        is_high = 1 if val > 120 else 0
        fields['fbs'] = {
            "value": is_high,
            "confidence": 0.85,
            "source": fbs_match.group(0)
        }
        
    return fields

def process_report(file_bytes: bytes, filename: str) -> dict:
    text = extract_text_from_file(file_bytes, filename)
    if not text.strip():
        return {
            "success": False,
            "error": "Could not extract text from the file.",
            "fields": {}
        }
        
    fields = parse_clinical_text(text)
    
    return {
        "success": True,
        "filename": filename,
        "extracted_fields": fields,
        "verification_required": list(fields.keys())
    }
