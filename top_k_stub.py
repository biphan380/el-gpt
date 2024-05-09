# TODO: Need to wrap below as a utility function for seeing which top_k nodes get retrieved. 

# The code below doesn't seem to be affecting the results that the index is returning below, 
# but are useful for inspecting which top k most relevant doc nodes are returned from a query
# can comment out once we see which top k nodes are returned

# query_str = '''You are an expert on human rights cases brought before the human rights tribunal of ontario. 
# I'm a court reporter at the Brampton Courthouse who was wrongfully dismissed. has there been a case 
# brought before the tribunal that's similar to my scenario? If so, give me the name of the case and summarize the case for me.'''
# query_embedding = embed_model.get_query_embedding(query_str)

# # query the vector store with dense search.
# from llama_index.core.vector_stores.types import (
# VectorStoreQuery,
# )
# query_obj = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5)

# from utils.to_file import write_query_results_to_file
# write_query_results_to_file(vector_store, query_obj, "topknodes.txt")