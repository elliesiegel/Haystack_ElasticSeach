# ElasticSeach with [Haystack](https://github.com/deepset-ai/haystack)

# What is [Haystack?](https://haystack.deepset.ai/docs/latest/intromd)

Search Engine Project for the LMU course about Search Engines.


### Recommended: Start Elasticsearch using Docker
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2

docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.11.1

if port occupied:

sudo lsof -i tcp:9200   :check weather important process, if not: 

sudo kill -9 PID        :where PID the process ID you want to kill 

--detach , -d		    :run container in background and print container ID

--publish , -p	        :publish a container's port(s) to the host

--env , -e		        :set environment variables


###  In Colab / No Docker environments: Start Elasticsearch from source

!wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q

!tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz

!chown -R daemon:daemon elasticsearch-7.9.2

```
import os

from subprocess import Popen, PIPE, STDOUT

def my_pre_exec():

    os.setegid(1000)

    os.seteuid(1000)
```


`es_server = Popen(['elasticsearch-7.9.2/bin/elasticsearch'],
                   stdout=PIPE, stderr=STDOUT,
                   preexec_fn=my_pre_exec
                  )`
            
Also instead of my_pre_exec, lambda: "preexec_fn=os.setuid(1)" as daemon. 
            
__wait until ES has started__

```!sleep 30```


### Preprocessing of documents

Haystack provides a customizable pipeline for:

- converting files into texts

- cleaning texts

- splitting texts

- writing them to a Document Store

Here: apply basic cleaning functions on texts, and index them in Elasticsearch.


`processor = PreProcessor(clean_empty_lines=True,
                         clean_whitespace=True,
                         clean_header_footer=True,
                         split_by="word",
                         split_length=200,
                         split_respect_sentence_boundary=True)`

