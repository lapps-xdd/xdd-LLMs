import inspect
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import PromptTemplate

entity_extraction_instruct_prompt = PromptTemplate.from_template(
    inspect.cleandoc(
        """
        What are the entities and their types from the following text? Entity type can be only one word. Generate the entity and its type as key-value pairs in a JSON object formatted as {{entity1: type1, entity2: type2, ...}}.
        Text: {text}
        """
    ))

summarization_instruct_prompt = PromptTemplate.from_template(
    inspect.cleandoc(
        """
        Generate a brief summary of the following text.
        Text: {text}
        """
    ))

if __name__ == '__main__':
    res = entity_extraction_instruct_prompt.invoke(
        {
            "text": "The dog is walking on the grass.",
        }
    )
    print(res.to_string())
