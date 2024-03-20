from langchain.llms import Ollama
from scripts import DATA_DIR
from scripts.load_data import load_json_text
import tqdm

from run_llm.construct_prompt import entity_extraction_instruct_prompt, summarization_instruct_prompt
import json

llm = Ollama(model="llama2", temperature=0, callback_manager=None)


def generate_entities():
    for topic_dir in DATA_DIR.iterdir():
        topic = topic_dir.name
        with open(f"{topic}_ollama_7b_entity.jsonl", "w") as out_f:
            for json_file in tqdm.tqdm(topic_dir.joinpath("output/doc").iterdir()):
                if json_file.suffix == ".json":
                    doc_id = json_file.name
                    text = load_json_text(json_file)
                    prompt = entity_extraction_instruct_prompt.invoke({"text": " ".join(text)}).to_string()
                    res = llm.invoke(prompt)
                    out_f.write(json.dumps({"doc_id": doc_id, "gpt_res": res}) + "\n")


def generate_summaries():
    for topic_dir in DATA_DIR.iterdir():
        topic = topic_dir.name
        with open(f"{topic}_ollama_7b_summarization.jsonl", "w") as out_f:
            for json_file in tqdm.tqdm(topic_dir.joinpath("output/doc").iterdir()):
                if json_file.suffix == ".json":
                    doc_id = json_file.name
                    text = load_json_text(json_file)
                    prompt = summarization_instruct_prompt.invoke({"text": " ".join(text)}).to_string()
                    res = llm.invoke(prompt)
                    out_f.write(json.dumps({"doc_id": doc_id, "gpt_res": res}) + "\n")


if __name__ == '__main__':
    generate_entities()
    generate_summaries()
