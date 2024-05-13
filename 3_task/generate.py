import sys

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PREFIX = 'get a title: '
MODEL_NAME = 'cointegrated/rut5-small'

if __name__ == '__main__':
    args = sys.argv
    path_to_article = args[1]
    prompt = PREFIX

    with open(path_to_article, 'r') as f:
        prompt += f.read().replace('\n', ' ')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained("./result_model").to(device)

    prompt_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    ).to(device)

    title_ids = model.generate(
        input_ids=prompt_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        max_length=25,
        num_beams=4,
    )

    decoded_text = tokenizer.decode(title_ids[0], skip_special_tokens=True)

    print(f'Generated title for article: {decoded_text}')
