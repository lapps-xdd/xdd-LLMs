import os
import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

from langchain_community.llms import Ollama
from tqdm import tqdm

from scripts import DATA_DIR
from scripts.load_data import load_json_text
from llm.construct_prompt import entity_extraction_instruct_prompt
from llm.construct_prompt import summarization_instruct_prompt


llm = Ollama(model="llama3", temperature=0, callback_manager=None)


def generate_entities():
    for topic_dir in DATA_DIR.iterdir():
        topic = topic_dir.name
        with open(f"{topic}_ollama_7b_entity.jsonl", "w") as out_f:
            for json_file in tqdm(topic_dir.joinpath("output/doc").iterdir()):
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
            for json_file in tqdm(topic_dir.joinpath("output/doc").iterdir()):
                if json_file.suffix == ".json":
                    doc_id = json_file.name
                    text = load_json_text(json_file)
                    prompt = summarization_instruct_prompt.invoke({"text": " ".join(text)}).to_string()
                    res = llm.invoke(prompt)
                    out_f.write(json.dumps({"doc_id": doc_id, "gpt_res": res}) + "\n")


def process_directory(doc_dir: str, sum_dir: str,
                      limit: int = sys.maxsize, overwrite: bool = False):
    """Process all json files in a directory and write text files with summaries
    to an output directory. This is like generate_summaries(), but integrates better
    with other xDD processing."""
    os.makedirs(sum_dir, exist_ok=True)
    print(f'\nProcessing {doc_dir}...')
    print(f'Writing to {sum_dir}...\n')
    docs = os.listdir(doc_dir)
    logfile = f'logs/processing-sum-{timestamp()}.txt'
    with open(logfile, 'w') as log:
        write_log_header(log, 'run_llm.run_ollama', doc_dir, [sum_dir], overwrite, limit)
        for doc in tqdm(list(Path(doc_dir).iterdir())):
            try:
                t0 = time.time()
                out_path = Path(sum_dir) / f'{doc.stem}.txt'
                if not overwrite and out_path.exists():
                    log.write(f'{doc.name}\talready exists\n')
                elif doc.suffix == '.json':
                    text = load_json_text(doc)
                    try:
                        summary = llm.invoke(prompt(text))
                        summary = trim_summary(summary)
                    except Exception:
                        log.write(f'{doc.name}\tfailed to invoke the LLM\n')
                        continue
                    with out_path.open('w') as fh_out:
                        fh_out.write(summary)
                    elapsed = time.time() - t0
                    log.write(f'{doc.name}\t{elapsed:.2f}\n')
            except Exception as e:
                log.write(f'{doc}\t{e}\n')


def timestamp():
    return datetime.strftime(datetime.now(), '%Y%m%d:%H%M%S')


def prompt(text: str):
    return summarization_instruct_prompt.invoke({"text": " ".join(text)}).to_string()


def trim_summary(summary: str):
    if '\n\n' in summary:
        parts = summary.split('\n\n')
        if len(parts) == 2:
            if parts[0].startswith('Here is a brief summary of the text:'):
                return parts[1]
    return summary


def write_log_header(log, command: str, indir: str, outdirs: list,
                     overwrite: bool, limit: int):
    log.write(f'# SCRIPT     =  run_llm.run_ollama\n')
    log.write(f'# INPUT      =  {indir}\n')
    for outdir in outdirs:
        log.write(f'# OUTPUT     =  {outdir}\n')
    log.write(f'# OVERWRITE  =  {str(overwrite)}\n')
    log.write(f'# LIMIT:     =  {limit}\n\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Run the llama3 summarizer over xDD files')
    parser.add_argument('--doc', help="directory with document structure parses")
    parser.add_argument('--sum', help="output directory for summarized data")
    parser.add_argument('--limit', help="Maximum number of documents to process",
                        type=int, default=sys.maxsize)
    parser.add_argument('--overwrite', help="Overwrite prior output", action='store_true')
    return parser.parse_args()



if __name__ == '__main__':

    if len(sys.argv) > 1:
        args = parse_args()
        process_directory(args.doc, args.sum, args.limit, args.overwrite)
    else:
        generate_entities()
        generate_summaries()
