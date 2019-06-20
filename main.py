from database_creation.database import Database
from database_creation.article import Article
from database_creation.sentence import Sentence
from database_creation.coreference import Coreference
from database_creation.np import Np
from database_creation.token import Token

# region Display parameters
Database.set_parameters(to_print=['articles'],
                        print_attribute=True,
                        random_print=True,
                        limit_print=10)

Article.set_parameters(to_print=['entities', 'title', 'date', 'abstract' 'contexts'],
                       print_attribute=True)

Coreference.set_parameters(to_print=['representative', 'entity'],
                           print_attribute=False)

Sentence.set_parameters(to_print=['text'],
                        print_attribute=False)

Np.set_parameters(to_print=['tokens'],
                  print_attribute=False)

Token.set_parameters(to_print=['word'],
                     print_attribute=False)
# endregion

# Parameters
max_size = 10000
min_articles = 2
min_queries = 3
n_queries = 3

# Initializes the database
database = Database(max_size=max_size)
database.preprocess_database()
database.filter(min_articles=min_articles)
database.preprocess_articles()
database.filter(min_queries=min_queries)
database.process_wikipedia(load=False)
database.create_task(load=False)

# Run the annotation task
Database.set_verbose(False)
database = Database(max_size=max_size, min_articles=min_articles, min_queries=min_queries)
database.create_task(load=True)
database.ask(n_queries)
database.gather()
