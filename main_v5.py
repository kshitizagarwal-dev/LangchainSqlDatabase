# import logging
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
import re
import ast
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


#langchain.debug = True

# Note: This model requires a large GPU, e.g. an 80GB A100. See documentation for other ways to run private non-OpenAI models.

# offline model
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
os.environ["OPENAI_API_KEY"] = "<open ai Key>"
openai.api_key = os.environ["OPENAI_API_KEY"]
llm_online = OpenAI(temperature=0, verbose=True)


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
    input = example['input']
    sql_cmd = example['sql_cmd'].replace("\\", " ")
    sql_result = example['sql_result']
    sql_answer = example['answer']
    table_info = db.table_info

      # Provided text  
    input_text = f"""\n- input: {input}
  table_info: |
    {table_info}
  sql_cmd: {sql_cmd}
  sql_result: "{sql_result}"
  answer: {sql_answer}
"""
    output_yaml_file_path = 'logs.yaml'
    with open(output_yaml_file_path, 'a') as file: 
        file.write(input_text)    

   
def perform_query_offline(QUERY):  
    example: any
    try:
        result = offline_chain(QUERY)
        print("*** Query succeeded")
        # 30th August 
        pattern = r"SQLResult: \[(.*?)\]\nAnswer"
        sql_results=result['intermediate_steps'][0]['input']
        match = re.search(pattern, sql_results)
        pattern2 = r"SQLResult: \nAnswer"
        match2 = re.search(pattern2, sql_results)
        #Handle blank SQLResult
        # if match2:
        #     result1 = perform_query_online(QUERY)
        #     example = _parse_example(result)
        #     parsing(example)
        #     return {'result':"I don't have answer to your query. Please re-phrase and try again!","intermediate_steps":"return Blank"} 
        has_none = False
        #Handle None SQLResult
        if match:
            desired_string = match.group(1)
            tuple_object = ast.literal_eval(desired_string)
            print(tuple_object)
            if len(tuple_object)==1:
               has_none = any(item is None for item in tuple_object)
            #print("Check NONE : ",has_none)
        if has_none:
            result1 = perform_query_online(QUERY)
            example = _parse_example(result1)
            # print("EXample: ",example)
            parsing(example)
            return {'result':"I don't have answer to your query. Please re-phrase and try again!","intermediate_steps":"return None"}
        #Handle result
        if result['result']=='' and  sql_results!='' and match:
            desired_string = match.group(1)
            return {'result':desired_string,"intermediate_steps":"Unable to generate Answer"}
        return result
    except Exception as exc:
        print("*** Query failed")
        result1 = perform_query_online(QUERY)
        example = _parse_example(result1)
        # print("EXample: ",example)
        parsing(example)
        return {'result':"I don't have answer to your query. Please re-phrase and try again!","intermediate_steps":"offline Query Failed"}
    

def perform_query_online(QUERY):
    try: 
        result = db_chain_online(QUERY)
        return result
    except Exception as exc:
        print("*** Online Query failed")
        result_ = {
                "query": QUERY,
                "intermediate_steps": exc.intermediate_steps
            }
        example = _parse_example(result_)
        # print("EXample: ",example)
        parsing(example)
        return {'result':"I don't have answer to your query. Please re-phrase and try again!","intermediate_steps":"Online Query Failed"}


example_prompt = PromptTemplate(
    input_variables=["table_info", "input", "sql_cmd", "sql_result", "answer"],
    template="{table_info}\n\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {sql_result}\nAnswer: {answer}",
)

with open("feedback.yaml", "r") as yaml_file:
    examples_dict = yaml.safe_load(yaml_file)

local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

example_selector = SemanticSimilarityExampleSelector.from_examples(
                        # This is the list of examples available to select from.
                        examples_dict,
                        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
                        local_embeddings,
                        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
                        Chroma,  # type: ignore
                        # This is the number of examples to produce and include per prompt
                        k=min(7, len(examples_dict)),
                    )
print("Example Selector: ", example_selector)
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=_sqlite_prompt + " Consider renewable energy same as green energy.\nHere are few examples:",
    suffix=PROMPT_SUFFIX,
    input_variables=["table_info", "input", "top_k"],
)
example_selector_online = SemanticSimilarityExampleSelector.from_examples(
                        # This is the list of examples available to select from.
                        examples_dict,
                        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
                        local_embeddings,
                        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
                        Chroma,  # type: ignore
                        # This is the number of examples to produce and include per prompt
                        k=min(5, len(examples_dict)),
                    )
few_shot_prompt_online = FewShotPromptTemplate(
    example_selector=example_selector_online,
    example_prompt=example_prompt,
    prefix=_sqlite_prompt + "\nHere are few examples:",
    suffix=PROMPT_SUFFIX,
    input_variables=["table_info", "input", "top_k"],
)



#offline
offline_chain = SQLDatabaseChain.from_llm(local_llm, db, prompt=few_shot_prompt, use_query_checker=True, verbose=True, return_intermediate_steps=True, top_k = 5)

#online
db_chain_online = SQLDatabaseChain.from_llm(llm_online, db,verbose=True, prompt = few_shot_prompt_online, return_intermediate_steps=True)


class Query(BaseModel):
    search_query: str



@app.post("/user_query_offline")
async def user_query_offline_route(query: Query):
      print("Mode : Offline****")
      query_response = perform_query_offline(query.search_query)
      steps = query_response["intermediate_steps"]
      final_result=query_response["result"]
      final_result=re.sub(r'Question:.*', '', final_result)
      final_result=re.sub(r'SQLQuery:.*', '', final_result)
      return {'result':final_result,'steps':steps}
       
@app.post("/user_query_online")
async def user_query_online_route(query: Query):
    print("Mode : Online****")
    query_response = perform_query_online(query.search_query)
    steps = query_response["intermediate_steps"]
    return {'result':query_response["result"],'steps':steps}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
