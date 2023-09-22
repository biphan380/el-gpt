from llama_index import VectorStoreIndex, download_loader

# SimpleWebPageReader = download_loader("SimpleWebPageReader")
from llama_index import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_dir="cases/"
)

documents = reader.load_data()
    
with open('documents.txt', 'w') as file:
    for doc in documents:
        # Write page_content
        file.write("Page Content:\n")
        file.write(doc.get_content())
        file.write("\n\n")

        # Write metadata
        file.write("Metadata:\n")
        for key, value in doc.metadata.items():
            file.write(f"{key}: {value}\n")

        # Add a separator between documents
        file.write("\n" + "-"*50 + "\n\n")

from llama_index.text_splitter import SentenceSplitter

print(len(documents))
# the reader loaded the entire case, but each page of the case was saved as a single document object, with the 
# page number and document name being the metadata. 
# curious to see the consequence of this on the semantic search engine.



# Here we import our SentenceSplitter to split document texts into smaller chunks, while
# preserving paragraphs/sentences as much as possible.

text_splitter = SentenceSplitter(
    chunk_size=1024,
    # separator=" ", ...no separator??
)

text_chunks = []

# maintain relationship with source doc index, to help inject doc metadata in the next step below

doc_idxs = []
for doc_idx, doc in enumerate(documents):
    print(f"currently at doc_idx number: {doc_idx}")
    cur_text_chunks = text_splitter.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

# Write text_chunks to a file
with open('text_chunks.txt', 'w') as chunk_file:
    for chunk in text_chunks:
        chunk_file.write(chunk)
        chunk_file.write("\n" + "seperator_starts@@@@@@@@@@@@@@@@@@@@@@seperator_ends"*50 + "\n")  # Separator between chunks for better readability


'''
We convert each chunk into a TextNode object, a low-level data abstraction in LlamaIndex 
that stores content but also allows defining metadata + relationships with other Nodes.

We inject metadata from the document into each node.
This essentially replicates logic in our SimpleNodePraser
'''

from llama_index.schema import TextNode

nodes = [] 

for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]] # We can see why doc_idxs is structured the way it is, to ensure
    node.metadata = src_doc.metadata   # That the node will always find which src_doc it came from.  
    nodes.append(node)                 # Only thing is, I don't think the src_doc has any metadata right now
                                       # Look at documents.txt and you will see the Metadata output is blank 

# print a sample node
with open('sample_node.txt', 'w') as node_file:
    for node in nodes:
        node_file.write(str(node))
        node_file.write("seperator_starts@@@@@@@@@@@@@@@@@@@@@@@@seperator_ends")

print(len(nodes))

# WARNING for Kent. At this point, I've deleted all the .txt files and re run the program many times, so 
# the txt files you're seeing may be different than what's shown in the previous commit 

# Let's extract metadata from the body of each node and attach it as metadata

from llama_index.node_parser.extractors import (
    MetadataExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-4")

metadata_extractor = MetadataExtractor(
    extractors=[
        TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=3, llm=llm),
    ],
    in_place=False,
)

# # nodes = metadata_extractor.process_nodes(nodes)

from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

# Create ChromaVectorStore

from llama_index.vector_stores import ChromaVectorStore, VectorStoreQuery, VectorStoreQueryResult
import chromadb

# create client and a new collection

import chromadb
db = chromadb.PersistentClient(path="./chroma_db")

collection = db.get_or_create_collection(name="my_collection")

vector_store = ChromaVectorStore(chroma_collection=collection)
vector_store.add(nodes=nodes)

# Query Data
# query_engine = index.as_query_engine()
# response = query_engine.query("Give me a summary of the Human Rights Tribunal Case between Ronald Phipps and the Toronto Police Services Board")
# print(response)

# from dataclasses import fields

# field_info = {f.name: f.type for f in fields(VectorStoreQuery)}
# print(field_info)

# field_info_result = {f.name: f.type for f in fields(VectorStoreQueryResult)}
# print(field_info_result)

from vector_store.vector_store_3b import VectorStore3B
vector_store = VectorStore3B()
# load nodes created from the two cases into the vector stores
vector_store.add(nodes)

query_str = '''

You are an expert on human rights cases brought before the human rights tribunal of ontario. Find a case that deals with family status and 
                give me the name of the case.'''
query_embedding = embed_model.get_query_embedding(query_str)

# query the vector store with dense search.

query_obj = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=1)

query_result = vector_store.query(query_obj)
# for similarity, node in zip(query_result.similarities, query_result.nodes):
#     print(
#         "\n----------------\n"
#         f"[Node ID {node.node_id}] Similarity: {similarity}\n\n"
#         f"{node.get_content(metadata_mode='all')}"
#         "\n----------------\n\n"
#     )

from llama_index import VectorStoreIndex
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine()
query_str = '''You are an expert on human rights cases brought before the human rights tribunal of ontario. Find a case that deals with family status and 
                give me the name of the case.
                '''
response = query_engine.query(query_str)
print(str(response))