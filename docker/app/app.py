from lib.prompt import Prompt
from lib.generator import generate
from fastapi import FastAPI
import uvicorn 


app = FastAPI()


@app.post("/generate/")
async def generate(Prompt: prompt) -> Prompt:
    prompt_text = prompt.get("prompt")
    print(prompt_text)
    gen = await generate(prompt)
    return {"response": gen}

if __name__ == "__main__":
   uvicorn.run("main:app", host="0.0.0.0", port=5600, reload=True)
