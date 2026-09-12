from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws")
async def audio_echo(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            byte = await websocket.receive_bytes()
            await websocket.send_bytes(byte)
            
    except WebSocketDisconnect:
        pass