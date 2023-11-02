

from llama_index import VectorStoreIndex
index = VectorStoreIndex.from_vector_store(vector_store)
retriever = index.as_retriever(similarity_top_k=1)
retrieved_nodes = retriever.retrieve(query_str)




from llama_index.storage import StorageContext
index.storage_context.persist(persist_dir="storage")

query_engine = index.as_query_engine()

# response = query_engine.query(query_str)
# print(str(response))