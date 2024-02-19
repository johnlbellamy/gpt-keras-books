from lib.prompt import Prompt
from lib.llm import generate
from lib.llm import transform
from fastapi import FastApi

app = FastAPI()


@app.post("/generate/")
async def generate(Prompt: prompt) -> Item:
    prompt_text = prompt.get("prompt")
