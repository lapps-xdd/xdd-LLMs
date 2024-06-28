from scripts import DATA_DIR
from scripts.load_data import load_json_text
import tqdm

from llm.construct_prompt import entity_extraction_instruct_prompt, summarization_instruct_prompt
from openai import OpenAI
import json

client = OpenAI()

COMPLETION_PARAMS = {
    "model": "gpt-4",  # or "gpt-3.5-turbo"
    "temperature": 0,
    "max_tokens": 500,  # max tokens in the output
    "messages": [],
}


def generate_message(prompt_str: str):
    return [{"role": "user", "content": prompt_str}]


def run_gpt(prompt_str: str):
    COMPLETION_PARAMS["messages"] = generate_message(prompt_str)
    response = client.chat.completions.create(**COMPLETION_PARAMS)
    content = response.choices[0].message.content
    return content


def generate_entities():
    for topic_dir in DATA_DIR.iterdir():
        topic = topic_dir.name
        with open(f"{topic}_gpt4_entity.jsonl", "w") as out_f:
            for json_file in tqdm.tqdm(topic_dir.joinpath("output/doc").iterdir()):
                if json_file.suffix == ".json":
                    doc_id = json_file.name
                    text = load_json_text(json_file)
                    prompt = entity_extraction_instruct_prompt.invoke({"text": " ".join(text)}).to_string()
                    res = run_gpt(prompt)
                    out_f.write(json.dumps({"doc_id": doc_id, "gpt_res": res}) + "\n")


def generate_summaries():
    for topic_dir in DATA_DIR.iterdir():
        topic = topic_dir.name
        with open(f"{topic}_gpt4_summarization.jsonl", "w") as out_f:
            for json_file in tqdm.tqdm(topic_dir.joinpath("output/doc").iterdir()):
                if json_file.suffix == ".json":
                    doc_id = json_file.name
                    text = load_json_text(json_file)
                    prompt = summarization_instruct_prompt.invoke({"text": " ".join(text)}).to_string()
                    res = run_gpt(prompt)
                    out_f.write(json.dumps({"doc_id": doc_id, "gpt_res": res}) + "\n")


if __name__ == '__main__':
    generate_entities()
    generate_summaries()
