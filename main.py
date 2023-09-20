from llama_index import VectorStoreIndex, download_loader

SimpleWebPageReader = download_loader("SimpleWebPageReader")

loader = SimpleWebPageReader()
# Ronald Phipps and TPSB
documents = loader.load_data(urls=['https://www.canlii.org/en/on/onhrt/doc/2009/2009hrto877/2009hrto877.html'])

# Hunter versus MAG
documents2 = loader.load_data(urls=['https://www.canlii.org/en/on/onhrt/doc/2023/2023hrto1081/2023hrto1081.html'])

# combine the two list[Document]
documents.extend(documents2)

# documents contains the Document object for Ronald and Betty George
    
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

# Here we import our SentenceSplitter to split document texts into smaller chunks, while
# preserving paragraphs/sentences as much as possible.

text_splitter = SentenceSplitter(
    chunk_size=1024,
    # separator=" ", ...no separator??
)

text_chunks = []

# maintain relationship with source doc index, to help inject doc metadata in the next step below
# I'm not sure how this relationship will help at the moment 

'''
So we have two Document objects in our documents list.
If, for example, the Document at index 0 got chunked into 2 pieces, and the Document at index1 got chunked into 3 pieces

doc_idxs would look like this:
[0,0,1,1,1]

and text_chunks would simply be a list that contains all 5 chunks

Not sure how this helps with node creation right now, but you can check the outputs below for learning
'''

doc_idxs = []
for doc_idx, doc in enumerate(documents):
    print(f"currently at doc_idx number: {doc_idx}")
    cur_text_chunks = text_splitter.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

# Write doc_idxs to a file
with open('doc_idxs.txt', 'w') as idx_file:
    for idx in doc_idxs:
        idx_file.write(str(idx) + '\n')

# Write text_chunks to a file
with open('text_chunks.txt', 'w') as chunk_file:
    for chunk in text_chunks:
        chunk_file.write(chunk)
        chunk_file.write("\n" + "seperator_starts@@@@@@@@@@@@@@@@@@@@@@seperator_ends"*50 + "\n")  # Separator between chunks for better readability


print(doc_idxs)

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

llm = OpenAI(model="gpt-4.0")

metadata_extractor = MetadataExtractor(
    extractors=[
        TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=3, llm=llm),
    ],
    in_place=False,
)

# nodes = metadata_extractor.process_nodes(nodes)

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

index = VectorStoreIndex.from_vector_store(vector_store)

# Query Data
# query_engine = index.as_query_engine()
# response = query_engine.query("Give me a summary of the Human Rights Tribunal Case between Ronald Phipps and the Toronto Police Services Board")
# print(response)

from dataclasses import fields

field_info = {f.name: f.type for f in fields(VectorStoreQuery)}
print(field_info)

field_info_result = {f.name: f.type for f in fields(VectorStoreQueryResult)}
print(field_info_result)