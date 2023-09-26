from llama_index import VectorStoreIndex, download_loader

# SimpleWebPageReader = download_loader("SimpleWebPageReader")
from llama_index import SimpleDirectoryReader
from utils.new_reader import CustomDirectoryReader

reader = CustomDirectoryReader(
    input_dir="cases/"
)

documents = reader.load_data()

from utils.to_file import write_documents_to_file
write_documents_to_file(documents)
    
from llama_index.text_splitter import SentenceSplitter

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
This essentially replicates logic in our SimpleNodePraser
'''

from llama_index.schema import TextNode

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






# Not sure if our metadata extractor works at this point

from llama_index.node_parser.extractors import (
    MetadataExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.llms import OpenAI

# llm = OpenAI(model="gpt-4")

# metadata_extractor = MetadataExtractor(
#     extractors=[
#         TitleExtractor(nodes=5, llm=llm),
#         QuestionsAnsweredExtractor(questions=3, llm=llm),
#     ],
#     in_place=False,
# )

# # nodes = metadata_extractor.process_nodes(nodes)

from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

from vector_store.vector_store_3b import VectorStore3B
vector_store = VectorStore3B()
# load nodes created from the two cases into the vector stores
vector_store.add(nodes)

# The code below doesn't seem to be affecting the results that the index is returning below, 
# but are useful for inspecting which top k most relevant doc nodes are returned from a query
# query_str = '''

# You are an expert on human rights cases brought before the human rights tribunal of ontario. Find a case that deals with family status and 
#                 give me the name of the case.'''
# query_embedding = embed_model.get_query_embedding(query_str)

# query the vector store with dense search.

# query_obj = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=1)

# query_result = vector_store.query(query_obj)
# for similarity, node in zip(query_result.similarities, query_result.nodes):
#     print(
#         "\n----------------\n"
#         f"[Node ID {node.node_id}] Similarity: {similarity}\n\n"
#         f"{node.get_content(metadata_mode='all')}"
#         "\n----------------\n\n"
#     )

from llama_index.vector_stores import VectorStoreQuery, VectorStoreQueryResult


from llama_index import VectorStoreIndex
index = VectorStoreIndex.from_vector_store(vector_store)

from llama_index.storage import StorageContext
index.storage_context.persist(persist_dir="storage")



query_engine = index.as_query_engine()
query_str = '''You are an expert on human rights cases brought before the human rights tribunal of ontario. I think I was recently discriminated 
for a job promotion because I was too old and they gave the job to a younger candidate. Has there ever been a case before
the tribunal that's similar to my scenario? Give me the name of the case.
                '''
response = query_engine.query(query_str)
print(str(response))