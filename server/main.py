from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
app = FastAPI()

model = WhisperModel("base")
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
                print(f"buffer full! collected {len(buffer)} bytes")
                buffer = b""
            
    except WebSocketDisconnect:
        pass