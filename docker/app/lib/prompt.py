from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: dict = {}
