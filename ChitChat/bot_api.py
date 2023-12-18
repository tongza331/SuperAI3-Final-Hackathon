# ตัวอย่างของ Chatbot API
# run with
#   uvicorn --host 0.0.0.0 --reload --port 3000 bot_api:app

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from retrieval import retrieve_doc, doc_df

app = FastAPI()

@app.get("/chat")
async def echo(line: str):
    doc_cids = retrieve_doc(line)
    answer = ""
    return PlainTextResponse("คำตอบอยู่ใน doc {}".format(doc_cids[0]))      # None = QA with no answer