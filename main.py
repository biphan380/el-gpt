from utils.new_reader import CustomDirectoryReader

reader = CustomDirectoryReader(
    input_dir="cases/"
)

documents = reader.load_data()

from utils.to_file import write_documents_to_file
write_documents_to_file(documents)

from llama_index.core.node_parser import SentenceSplitter

# Here we import our SentenceSplitter to split document texts into smaller chunks, while
# preserving paragraphs/sentences as much as possible.

text_splitter = SentenceSplitter(
    chunk_size=1024,
    # separator=" ", ...no separator??
)

text_chunks = []

# maintain relationship with source doc index, to help inject doc metadata in the next step below

doc_indexes = []
for doc_index, doc in enumerate(documents):
    cur_text_chunks = text_splitter.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_indexes.extend([doc_index] * len(cur_text_chunks))

from utils.to_file import write_text_chunks_to_file

write_text_chunks_to_file(text_chunks)


'''
We convert each chunk into a TextNode object, a low-level data abstraction in LlamaIndex 
that stores content but also allows defining metadata + relationships with other Nodes.

We inject metadata from the document into each node.
This essentially replicates logic in our SimpleNodeParser
'''

from llama_index.core.schema import TextNode

nodes = [] 
# used 'i' to mean 'index' because in a lot of languages, the index position that the iterator returns (e.g enumerate) is declared as 'i'
for i, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_indexes[i]] # We can see why doc_idxs is structured the way it is, to ensure
    node.metadata = src_doc.metadata   # That the node will always find which src_doc it came from.  
    nodes.append(node)                 # Only thing is, I don't think the src_doc has any metadata right now
                                       # Look at documents.txt and you will see the Metadata output is blank 

from utils.to_file import write_nodes_to_file
write_nodes_to_file(nodes)

from llama_index.core.node_parser.extractors import (
    MetadataExtractor,
    QuestionsAnsweredExtractor,
    # TitleExtractor, NOTE: title extractor is currently broken
)
from llama_index.core.llms import OpenAI
from llama_index.core.llms.llama_utils import messages_to_prompt, completion_to_prompt

# re-inject llamaCPP 
llm = OpenAI(model="gpt-3.5-turbo")

metadata_extractor = MetadataExtractor(
    extractors=[
        # TitleExtractor(nodes=5, llm=llm), NOTE: title extractor is currently broken
        QuestionsAnsweredExtractor(questions=3, llm=llm),
    ],
    in_place=False,
)

import os
import pickle

'''
The code below with the cache_file is for caching the nodes that have already been processed so 
we don't need to re-process them every single time we run the program. This doesn't account for 
when we add another case to our /cases directory. 

We will develop a better caching mechanism, but that might involve caching the actual index/vector store that's created from the nodes, and not the nodes themselves.
For now, just delete the processed_nodes.pk1 file everytime we add more cases.
'''
# Define the path to the cache file 
cache_file = 'processed_nodes.pk1'

if os.path.exists(cache_file):
    # If cache file exists, load the processed nodes from the file 
    with open(cache_file, 'rb') as f:
        nodes = pickle.load(f)
else:
    # If cache file does not exist, process the nodes and save the result to the file
    nodes = nodes
    nodes = metadata_extractor.process_nodes(nodes)
    with open(cache_file, 'wb') as f:
        pickle.dump(nodes, f)

# print out the nodes with their new metadata 
write_nodes_to_file(nodes)



from llama_index.core.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

from vector_store.vector_store_3b import VectorStore3B
vector_store = VectorStore3B()
# load nodes created from the cases into the vector stores
vector_store.add(nodes)

query_str = '''You are an expert on human rights cases brought before the human rights tribunal of ontario. 
Provide a summary of the Betty George case.'''
query_embedding = embed_model.get_query_embedding(query_str)

from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from custom_retriever import CustomRetriever

retriever = CustomRetriever(
    vector_store, embed_model, query_mode = "default", similarity_top_k=2, query_str=query_str
)

retrieved_nodes = retriever.retrieve(query_str)

from llama_index.core.response.pprint_utils import pprint_source_node 

for node in retrieved_nodes:
    pprint_source_node(node, source_length=1000)

from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever)

response = query_engine.query(query_str)

print(str(response))













# '''
# NOTE: if top_k is set to more than 1, there's a chance the retrieved doc nodes
# will not all be from the same case, i.e., the most relevant case. This means
# the llm's response might hallucinate and give the case name of node with the 2nd or 3rd highest
# top_k score, which could be the wrong case.

# When we use a qa_prompt and set the top_k to 1, i.e., only give the most relevant node
# as context for the query string, the results are quite good.
# '''

# from llama_index.core.prompts import PromptTemplate

# qa_prompt = PromptTemplate(
#     """\
# Context information is below.
# ---------------------
# {context_str}
# ---------------------
# Given the context information and not prior knowledge, answer the query.
# Query: {query_str}
# Answer: \
# """ 
# )

# query_str = '''You are an expert on human rights cases brought before the human rights tribunal of ontario. 
# I'm a post man that was recently stopped and frisked by the police for being black. has there been a case 
# brought before the tribunal that's similar to my scenario? If so, give me the name of the case and summarize the case for me.'''

# from llama_index.core import VectorStoreIndex
# index = VectorStoreIndex.from_vector_store(vector_store)
# retriever = index.as_retriever(similarity_top_k=1)
# retrieved_nodes = retriever.retrieve(query_str)

# def generate_response(retrieved_nodes, query_str, qa_prompt, llm):
#     context_str = "\n\n".join([r.get_content() for r in retrieved_nodes])
#     fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
#     response = llm.complete(fmt_qa_prompt)
#     return str(response), fmt_qa_prompt

# response, fmt_qa_prompt = generate_response(retrieved_nodes, query_str, qa_prompt, llm)
# print(f"Response (k=1): {response}")

# from llama_index.core.storage import StorageContext
# index.storage_context.persist(persist_dir="storage")

# query_engine = index.as_query_engine()

# response = query_engine.query(query_str)
# print(str(response))
