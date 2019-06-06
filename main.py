# region Imports
from database_creation.database import Database
from database_creation.article import Article
from database_creation.sentence import Sentence
from database_creation.coreference import Coreference
from database_creation.np import Np
from database_creation.token import Token
# endregion

# region Display parameters
Database.set_parameters(to_print=['articles'],
                        print_attribute=True,
                        random_print=True,
                        limit_print=10)

Article.set_parameters(to_print=['entities', 'title', 'contexts'],
                       print_attribute=True)

Coreference.set_parameters(to_print=['entity', 'representative'],
                           print_attribute=True)

Sentence.set_parameters(to_print=['text'],
                        print_attribute=False)

Np.set_parameters(to_print=['tokens'],
                  print_attribute=False)

Token.set_parameters(to_print=['word'],
                     print_attribute=False)
# endregion

database = Database(max_size=10000)

database.preprocess_database()
database.stats_tuples()

database.filter_threshold(threshold=10)
database.preprocess_articles()

database.process_contexts()
database.stats_contexts()
