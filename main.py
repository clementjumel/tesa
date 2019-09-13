from database_creation.database import Database

max_size = 10000
shuffle = True
min_articles = 1
min_queries = 1

database = Database(max_size=max_size, shuffle=shuffle, min_articles=min_articles, min_queries=min_queries)

database.preprocess_database(debug=True)
database.process_articles(debug=True)

database.process_wikipedia(load=False, debug=True)
database.process_queries(load=False, debug=True)

# database.combine_pkl(in_names=[
#     'wikipedia_global',
#     'wikipedia_size10k_shuffle_articles1_queries1',
# ])
