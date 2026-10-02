from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import numpy as np
app = FastAPI()

model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper model loaded!")


@app.websocket("/ws")
async def audio_echo(websocket: WebSocket):
    await websocket.accept()
    buffer = b""
    try:
        while True:
            data = await websocket.receive_bytes()
            buffer += data
            if len(buffer)>= 192000 :
                samples = np.frombuffer(buffer, dtype=np.float32)
                downsampled = samples[::3]
                segments, info = model.transcribe(downsampled, language="en")
                for segment in segments:
                    print("The voice is:", segment.text)
                buffer = b""    
            
    except WebSocketDisconnect:
        pass