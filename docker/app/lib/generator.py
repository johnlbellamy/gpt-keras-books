from keras import Model
from lib.text_generator import TextGenerator


def generate(prompt: str, model: Model):
    print(f"Received: {prompt}")
    text_gen = TextGenerator(model=model,
                             k=15)
    prompt_clean = TextGenerator.remove_punctuation(prompt)
    prompt_clean = prompt_clean.replace('\n', "")
    prompt_clean = prompt_clean.replace("\n", "")
    prompt_clean = prompt_clean.lower()
    prompt_tokens = text_gen.get_start_tokens(prompt=prompt_clean)
    prompt_tokens = [x for x in prompt_tokens if x]
    gen_text = text_gen.generate(prompt_tokens=prompt_tokens)
    print("Got text!")
    return gen_text



