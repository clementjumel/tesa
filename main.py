from database_creation.database import create_queries, annotation_task, gather_answers

# Parameters
max_size = 10000
min_articles = 1
min_queries = 1
n_queries = 3

# Create the queries database
create_queries(max_size=max_size, min_articles=min_articles, min_queries=min_queries)

# Run the annotation task
annotation_task(n_queries=n_queries, max_size=max_size, min_articles=min_articles, min_queries=min_queries)

# Gather the answers
gather_answers(max_size=max_size, min_articles=min_articles, min_queries=min_queries)
