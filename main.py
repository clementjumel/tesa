from database_creation.database import Database

max_size = 10000
min_articles = 1
min_queries = 1

database = Database(max_size=max_size, min_articles=min_articles, min_queries=min_queries)

database.preprocess_database()
database.process_articles()

database.process_wikipedia(load=False)
database.process_queries(load=False)
