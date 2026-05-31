import json
import os
from tqdm import trange, tqdm
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


prompt_film = '''
Write a concise and informative description of the film {film_title} whose genre is {genre}, with consideration of the following aspects:

The brief plot overview (without major spoilers).
The main themes and genre.
The setting or historical context.
Key characters and the actors who portray them.
The film's director and any notable stylistic elements.
The overall tone and mood of the film.
Its cultural impact or deeper meaning, if applicable.

Keep the description under 80 words.

Please only return the description.
'''

prompt_movie = '''
Write a concise and informative description of the movie/TV {movie_title} whose genre is {genre}, with consideration of the following aspects:

The brief plot overview (without major spoilers).
The main themes and genre.
The setting or historical context.
Key characters and the actors who portray them.
The movie's director and any notable stylistic elements.
The overall tone and mood of the movie.
Its cultural impact or deeper meaning, if applicable.

Keep the description under 80 words.

Please only return the description.
'''

prompt_book = '''
Write a concise and informative description of the book {book_title}. ISBN of the book is {isbn}. The author of the book is {author}.
The publication year of the book is {year}. Its publisher is {publisher}. The description could include the following details:

- A brief plot overview (without major spoilers).
- The main themes and genre.
- The setting or historical context.
- Key characters and their roles.
- The author's name and any notable stylistic elements.
- The overall tone and style of the writing.
- Any cultural or literary impact, if applicable.
- Basic publication details of year and publisher.

Keep the description under 80 words."

Please only return the description.
'''


def get_textual_description(args):
    data_dir = f"../data/{args.dataset}/proc_data"

    # 用 HuggingFace Transformers 替代 vLLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    all_objs = {}

    if args.dataset == "ml-1m":
        movie_dict = json.load(open(os.path.join(data_dir, "movie_detail.json"), "r"))
        for i in range(1, 3953):
            key = str(i)
            if key not in movie_dict.keys():
                title, genre = "", ""
            else:
                title, genre = movie_dict[key]
            text = prompt_film.format(film_title=title, genre=genre)
            all_objs[key] = text

    elif args.dataset == "BookCrossing":
        id2book = json.load(open(os.path.join(data_dir, "id2book.json"), "r"))
        for i in trange(len(id2book)):
            key = str(i)
            isbn, title, author, year, publisher = id2book[key]
            text = prompt_book.format(book_title=title, isbn=isbn, author=author, year=year, publisher=publisher)
            all_objs[key] = text

    elif args.dataset == "amazon-movies":
        movie_dict = json.load(open(os.path.join(data_dir, "idx2movie.json"), "r"))
        for i in trange(len(movie_dict)):
            key = str(i)
            if key not in movie_dict.keys():
                title, genre = "", ""
            else:
                _, title, genre = movie_dict[key]
            text = prompt_movie.format(movie_title=title, genre=genre)
            all_objs[key] = text
    else:
        raise NotImplementedError

    decoding_objs = all_objs

    save_path = f"../data/{args.dataset}/proc_data"
    os.makedirs(save_path, exist_ok=True)
    output_file = save_path + '/{}_text.json'.format(args.dataset)

    write_objs = {}

    for i in tqdm(range(len(decoding_objs))):
        key = str(i)
        if key not in decoding_objs.keys():
            continue
        cur_obj = decoding_objs[key]

        messages = [
            {"role": "system", "content": "You are a friendly and helpful assistant who always responds to the query from users"},
            {"role": "user", "content": cur_obj},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                min_new_tokens=10,
                temperature=0.8,
                top_p=0.90,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        textual_description = generated_text.replace('</s>', '').replace('\n', '').replace('"', '')
        write_objs[key] = textual_description

    json.dump(write_objs, open(output_file, "w"))
    print(f"Done. Saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/mnt/cache/sichunluo2/Llama-3.1-8B-Instruct', help="")
    parser.add_argument('--dataset', type=str, default="ml-1m", help="ml-1m/BookCrossing/amazon-movies")
    args = parser.parse_args()
    get_textual_description(args)
