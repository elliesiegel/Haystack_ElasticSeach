import glob
import json
import time

from haystack.preprocessor.preprocessor import PreProcessor
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers


# log the time 
start_time = time.time()

# Recommended: Start Elasticsearch using Docker
# docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2
# docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.11.1

# if port occupied:
# sudo lsof -i tcp:9200 - check weather important process, if not: 
# sudo kill -9 PID      - where PID the process ID you want to kill 

# --detach , -d		Run container in background and print container ID
# --publish , -p	Publish a container's port(s) to the host
# --env , -e		Set environment variables


# # In Colab / No Docker environments: Start Elasticsearch from source
# !wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q
# !tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
# !chown -R daemon:daemon elasticsearch-7.9.2

# import os
# from subprocess import Popen, PIPE, STDOUT

# def my_pre_exec():
#     os.setegid(1000)
#     os.seteuid(1000)

# es_server = Popen(['elasticsearch-7.9.2/bin/elasticsearch'],
#                    stdout=PIPE, stderr=STDOUT,
#                    preexec_fn=my_pre_exec     # Also lambda: "preexec_fn=os.setuid(1)" as daemon. 
#                   )
# # wait until ES has started
# !sleep 30

# """ Preprocessing of documents
# Haystack provides a customizable pipeline for:
#  - converting files into texts
#  - cleaning texts
#  - splitting texts
#  - writing them to a Document Store

# Here: apply basic cleaning functions on texts, and index them in Elasticsearch.
# """

# processor = PreProcessor(clean_empty_lines=True,
#                          clean_whitespace=True,
#                          clean_header_footer=True,
#                          split_by="word",
#                          split_length=200,
#                          split_respect_sentence_boundary=True)


# Connect to Elasticsearch and prepare index for documents
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="jsons_letters")

# json_files_dir = sys.argv[1]
data_files = r"../suchmaschinen_briefe/letters/*/json/*.json"
letter_files = glob.glob(data_files)

for letter_file in letter_files:
    with open(letter_file, "r") as json_file:
        json_file = json.load(json_file)

        """1. Document Store: Haystack finds answers to queries within the documents stored in a DocumentStore. """

        """
        Start an Elasticsearch server:
            - You can start Elasticsearch on your local machine instance using Docker. 
            - If Docker is not readily available in your environment (eg., in Colab notebooks), 
            then you can manually download and execute Elasticsearch from source.
        """
        # Write the dicts containing documents to the DB.
        # document_store.write_documents([json_file])


""" Retriever

Retrievers help narrowing down the scope for the Reader to smaller units of text where a given question could be answered.
Here: Elasticsearch's default BM25 algorithm 
"""

# from haystack.retriever.sparse import ElasticsearchRetriever
# retriever = ElasticsearchRetriever(document_store=document_store)

# Alternative: An in-memory TfidfRetriever based on Pandas dataframes for building quick-prototypes with SQLite document store.
from haystack.retriever.sparse import TfidfRetriever
retriever = TfidfRetriever(document_store=document_store)


""" Reader

A Reader scans the texts returned by retrievers in detail and extracts the k best answers. They are based on powerful, 
but slower deep learning models. Haystack currently supports Readers based on the frameworks FARM and Transformers.

Hint: You can adjust the model to return "no answer possible" with the no_ans_boost. Higher values mean the model prefers "no answer possible"
"""

# FARMReader
# Load a  local model or any of the QA models on
# Hugging Face's model hub (https://huggingface.co/models)

reader = FARMReader(model_name_or_path="bert-base-multilingual-cased", use_gpu=True)

# Alternative: TransformersReader
# reader = TransformersReader(model_name_or_path="bert-base-multilingual-cased", tokenizer="distilbert-base-uncased", use_gpu=-1)


""" Pipeline

stick together building blocks to a search pipeline. Pipelines are Directed Acyclic Graphs (DAGs).
A few predefined Pipelines available. 
One of them is the ExtractiveQAPipeline that combines a retriever and a reader to answer questions.
Learn more about Pipelines in the [docs](https://haystack.deepset.ai/docs/latest/pipelinesmd).
"""

from haystack.pipeline import ExtractiveQAPipeline
pipe = ExtractiveQAPipeline(reader, retriever)


"""Ask a question"""
# You can configure how many candidates the reader and retriever shall return
# The higher top_k_retriever, the better (but also the slower) the answers. 
# Ex.: prediction = pipe.run(query="Ist mein Fahrrand mitversichert?", top_k_reader=5)

question = "Whom met James Joyce at the Liberal Club on 25 January 1903?"
# correct answer: "I met Archer at the Liberal Club but our talk though it lasted a long time was not very business-like. 
# I also met Lady Gregory and had just time to see Mr O’Connell before I caught my train."

prediction = pipe.run(query=question, top_k_retriever=5, top_k_reader=5)

print()
print("Question: ", question)
print()
print_answers(prediction, details="minimal")
print()

print("--- %s seconds ---" % (time.time() - start_time))
# on average:  --- 9.938042879104614 seconds ---
