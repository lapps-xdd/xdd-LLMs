# xdd-LLMs

### Running Instructions
Put your data in `data/` folder formatted as this:
```
data
├── biomedical-0100
├── climate-change-modeling-0100
├── covid-19-0100
├── cultivars-0100
├── geoarchive-0100
├── mars-0100
└── molecular-physics-0100
```
#### Data Preparation

#### Summarization and Entity Extraction with GPT
in `run_llm/run_gpt.py` file.
```shell
python -m run_llm.run_gpt
```
- You need an [OpenAI API key](https://openai.com/) to run this code.
- You can switch between GPT-3.5 and GPT-4 by configuring the Line 12


#### Summarization and Entity Extraction with LLama2
```shell
python -m run_llm.run_ollama
```
- You need to download LLama2 checkpoint using [Ollama](https://github.com/ollama/ollama) to run this code. 
