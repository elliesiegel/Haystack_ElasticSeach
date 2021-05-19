import argparse
import glob
import json
import time

# from haystack.preprocessor.preprocessor import PreProcessor
from haystack.reader.farm import FARMReader
# from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers

# call: python3 haystack_qa_pipeline.py
# --to_index (optional)

# log the time 
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--to_index",
    action='store_true',
    help="if given, index the document in ES"
)

args = parser.parse_args()
to_index = args.to_index


def index_docs(letter_files):

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
            document_store.write_documents([json_file])


# Connect to Elasticsearch and prepare index for documents
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="jsons_letters")


# json_files_dir = sys.argv[1]
data_files = r"../suchmaschinen_briefe/letters/*/json/*.json"
letter_files = glob.glob(data_files)

if to_index:
    index_docs(letter_files)


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

# -------------------------------------------------------
question = "At what school did James Joyce teach?"
# "I am an English teacher here in a Berlitz School. 
# I have been here for sixteen months during which time I have achieved the delicate task of 
# living and of supporting two other trusting souls on a salary of £80 a year."
# haystack answer: It seems we can’t spend Xmas together. All good wishes from '
                #  'us all. Thanks for yours

# question = "Who offered Virginia Woolf a black kitten?"
# correct answer: I have been offered it by Miss Power
# haystack answer:  'answer': 'disposition (like mine!) I have been offered it by Miss '
                  # 'Power, but my room apparently would not suit it, as it '
                  # 'ought'

# question = "Der Kampf ums Dasein heißt nach Freud?"
# correct answer: Der ›Kampf ums Dasein‹ heißt für mich noch ein Kampf ums ›Dableiben‹

# question = "Whom met James Joyce at the Liberal Club on 25 January 1903?"
# correct answer: "I met Archer at the Liberal Club but our talk though it lasted a long time was not very business-like. 
# I also met Lady Gregory and had just time to see Mr O’Connell before I caught my train."

# -------------------------------------------------------

prediction = pipe.run(query=question, top_k_retriever=5, top_k_reader=5)

print()
print("Question: ", question)
print()
print_answers(prediction, details="minimal")
print()

print("--- %s seconds ---" % (time.time() - start_time))
# overall run time on average:  --- between 10 and 20 seconds ---
