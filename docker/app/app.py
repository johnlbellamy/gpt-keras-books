from lib.prompt import Prompt
from lib.generator import generate
from fastapi import FastAPI
import uvicorn
from keras import Model
from lib.gpt import build_gpt

app = FastAPI()
model: Model


@app.post("/generate/")
async def generate_text(prompt: Prompt) -> dict:
    gen = generate(prompt=prompt.prompt, model=model)
    return {"response": gen}


@app.on_event("startup")
def on_startup():
    global model
    model = build_gpt()
    model.load_weights("/serving/app/bin/gpt-books.weights.h5")
    model.summary()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5600, reload=True)
