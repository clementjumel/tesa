from database_creation.database import Database

max_size = 10000
shuffle = True
min_articles = 1
min_queries = 1
random_seed = 0

database = Database(max_size=max_size, shuffle=shuffle,
                    min_articles=min_articles, min_queries=min_queries,
                    random_seed=random_seed)

database.preprocess_database(debug=True)
database.process_articles(debug=True)

database.process_wikipedia(load=True, debug=True)
database.process_queries(load=False, check_changes=True, debug=True)

database.combine_pkl()
