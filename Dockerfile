FROM python:3.12
WORKDIR /usr/shr/app
RUN apt-get update && apt-get install -y libgl1

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Установить FastAPI и uvicorn
RUN pip install fastapi uvicorn
# RUN apt-get install -y ffmpeg

COPY . .

# Запуск FastAPI приложения
 CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# RUN pip install debugpy

# CMD ["debugpy", "--listen", "0.0.0.0:5678", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


