from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from inference import process_video

app = FastAPI()

INPUT_FOLDER = "videos"
OUTPUT_FOLDER = "results"

class VideoRequest(BaseModel):
    video_name: str

@app.post("/process_video")
async def process_video_api(request: VideoRequest):
    video_path = request.video_name
    
    # Проверка наличия видео
    if not os.path.exists(os.path.join(INPUT_FOLDER, request.video_name)):
        raise HTTPException(status_code=404, detail="Нет такого видео, пожалуйста загрузите видео в папку videos")
    
    # Обработка видео
    try:
        print(f"Path to video: {video_path}")
        process_video(video_path, show_video=False, save_video=True, show_plate=True)
        return {"message": f"Готово, проверьте папку {OUTPUT_FOLDER}/{request.video_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки видео: {str(e)}")
