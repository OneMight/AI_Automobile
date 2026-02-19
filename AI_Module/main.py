from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from classes import RecognitionResponse 
from recognize_car import get_prediction_data
import os
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=RecognitionResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    result = get_prediction_data(image_bytes)
    
    if result is None:
        raise HTTPException(status_code=404, detail="Автомобиль не распознан на фото")

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=os.environ.get("$PORT") | 8000)