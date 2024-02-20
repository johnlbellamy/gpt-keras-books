from lib.text_generator import TextGenerator



model = build_gpt()
model.load_weights("gpt-books.weights.h5")
model.summary()

TEXT_GEN = TextGenerator(model=model,
                         max_tokens=10)

async def generate(prompt):
    prompt_tokens = TEXT_GEN.get_start_tokens(prompt=prompt)
    prompt_tokens = [x for x in prompt_tokens if x ]
    return TEXT_GEN.generate(prompt_tokens=prompt_tokens)
