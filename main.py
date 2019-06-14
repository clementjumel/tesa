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

database = Database()

database.preprocess_database()
database.filter_threshold(threshold=5)

database.preprocess_articles()
database.process_contexts()

database.process_wikipedia(load=False)

database.process_samples(load=False)
