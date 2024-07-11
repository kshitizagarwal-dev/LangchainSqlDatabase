import logging
import torch
from transformers import AutoTokenizer, GPT2TokenizerFast, pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from typing import Dict
import yaml
from langchain.llms import OpenAI
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.chains.sql_database.prompt import _sqlite_prompt, PROMPT_SUFFIX
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.example_selector.semantic_similarity import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import openai
import langchain
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# langchain.debug = True

# Note: This model requires a large GPU, e.g. an 80GB A100. See documentation for other ways to run private non-OpenAI models.


def initialize_llm():
    model_id = "HuggingFaceH4/starchat-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", tokenizer=tokenizer, model=model_id ,torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=1024,temperature=0, eos_token_id=49155)
    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Offline Model
local_llm = initialize_llm()

table_name_list = ["energy_data","energy_supply"]
db = SQLDatabase.from_uri("sqlite:///./energydata.db", include_tables=table_name_list)

#Online Model
os.environ["OPENAI_API_KEY"] = "<open-ai key"
openai.api_key = os.environ["OPENAI_API_KEY"]
llm_online = OpenAI(temperature=0, verbose=True)
db_chain_online = SQLDatabaseChain.from_llm(llm_online, db,verbose=True, return_intermediate_steps=True)



def _parse_example(result: Dict) -> Dict:
    sql_cmd_key = "sql_cmd"
    sql_result_key = "sql_result"
    table_info_key = "table_info"
    input_key = "input"
    final_answer_key = "answer"

    _example = {
        "input": result.get("query"),
    }

    steps = result.get("intermediate_steps")
    answer_key = sql_cmd_key # the first one
    for step in steps:
        # The steps are in pairs, a dict (input) followed by a string (output).
        # Unfortunately there is no schema but you can look at the input key of the
        # dict to see what the output is supposed to be
        if isinstance(step, dict):
            # Grab the table info from input dicts in the intermediate steps once
            if table_info_key not in _example:
                _example[table_info_key] = step.get(table_info_key)

            if input_key in step:
                if step[input_key].endswith("SQLQuery:"):
                    answer_key = sql_cmd_key # this is the SQL generation input
                if step[input_key].endswith("Answer:"):
                    answer_key = final_answer_key # this is the final answer input
            elif sql_cmd_key in step:
                _example[sql_cmd_key] = step[sql_cmd_key]
                answer_key = sql_result_key # this is SQL execution input
        elif isinstance(step, str):
            # The preceding element should have set the answer_key
            _example[answer_key] = step
    return _example


def parsing(example):
    input = example['input'].search_query
    sql_cmd = example['sql_cmd'].replace("\\", " ")
    sql_result = example['sql_result']
    sql_answer = example['answer']

      # Provided text  
    input_text = f"""\n- input: {input}
  table_info: |
    CREATE TABLE energy_data (
    "ISO" TEXT,
    "Energy" TEXT,
    "Units" TEXT,
    "Month" TEXT,
    "Year" INTEGER,
    "Generation" REAL
    )
    /* 3 rows from energy_data table:
      ISO     Energy           Units               Month    Year    Generation
      ISO-NE  Natural gas      billion kilowatthours  January  2011    4.43
      ISO-NE  Coal             billion kilowatthours  January  2011    1.38
      ISO-NE  Nuclear          billion kilowatthours  January  2011    3.47 */

    CREATE TABLE energy_supply (
      "ISO" TEXT,
      "Units" TEXT,
      "Month" TEXT,
      "Year" INTEGER,
      "Demand" REAL,
      "Generation" REAL
    )
    /* 3 rows from energy_supply table:
      ISO      Units                   Month   Year    Demand  Generation
      CAISO     billion kilowatthours    April    2011    20.8    15.05
      CAISO     billion kilowatthours    April    2012    21.1    13.34
      CAISO     billion kilowatthours    April    2013    21.5    13.53*/
  sql_cmd: {sql_cmd}
  sql_result: "{sql_result}"
  answer: {sql_answer}"""
    output_yaml_file_path = 'logs.yaml'
    with open(output_yaml_file_path, 'a') as file: 
        file.write(input_text)    

   
def perform_query_offline(QUERY):  
    # example: any
    # try:
    #     result = offline_chain(QUERY)
    #     print("*** Query succeeded")
    #     # example = _parse_example(result)
    # except Exception as exc:
    #     print("*** Query failed")
    #     result = perform_query_online(db_chain_online, QUERY)
    #     # result = {
    #     #     "query": QUERY,
    #     #     "intermediate_steps": exc.intermediate_steps
    #     # }
    #     example = _parse_example(result)
    #     print("EXample: ",example)
    #     parsing(example)
    result = offline_chain(QUERY)
    return result

def perform_query_online(db_chain_online, QUERY):
    result = db_chain_online(QUERY)
    return result


example_prompt = PromptTemplate(
    input_variables=["table_info", "input", "sql_cmd", "sql_result", "answer"],
    template="{table_info}\n\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {sql_result}\nAnswer: {answer}",
)

with open("output_file.yaml", "r") as yaml_file:
    examples_dict = yaml.safe_load(yaml_file)

# examples_dict = yaml.safe_load(data)
#print("Example :", examples_dict)

local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

example_selector = SemanticSimilarityExampleSelector.from_examples(
                        # This is the list of examples available to select from.
                        examples_dict,
                        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
                        local_embeddings,
                        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
                        Chroma,  # type: ignore
                        # This is the number of examples to produce and include per prompt
                        k=min(3, len(examples_dict)),
                    )

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=_sqlite_prompt + "Here are some examples:",
    suffix=PROMPT_SUFFIX,
    input_variables=["table_info", "input", "top_k"],
)

offline_chain = SQLDatabaseChain.from_llm(local_llm, db, prompt=few_shot_prompt, use_query_checker=True, verbose=True, return_intermediate_steps=True)


class Query(BaseModel):
    search_query: str

@app.post("/user_query_offline")
async def user_query_offline_route(query: Query):

      query_response = perform_query_offline(query)
      steps = query_response["intermediate_steps"]
      return {'result':query_response,'steps':steps}
       
@app.post("/user_query_online")
async def user_query_online_route(query: Query):
    query_response = perform_query_online(db_chain_online, query)
    return {'result':query_response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
