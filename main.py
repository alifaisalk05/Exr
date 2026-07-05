import io
import easyocr
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="EasyOCR API")

# Load model once
reader = easyocr.Reader(["en"], gpu=False)

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "EasyOCR API is running"
    }

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = np.array(image)

        result = reader.readtext(image)

        text = " ".join([r[1] for r in result])

        return JSONResponse({
            "success": True,
            "text": text,
            "results": result
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
