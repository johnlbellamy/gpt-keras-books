from keras import Model
from lib.text_generator import TextGenerator


def generate(prompt: str, model: Model):
    print(f"Received: {prompt}")
    text_gen = TextGenerator(model=model,
                             max_tokens=10)
    prompt_tokens = text_gen.get_start_tokens(prompt=prompt)
    prompt_tokens = [x for x in prompt_tokens if x]
    gen_text = text_gen.generate(prompt_tokens=prompt_tokens)
    print("Got text!")
    return gen_text
