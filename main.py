# region Imports
from database_creation.database import Database
from database_creation.article import Article
from database_creation.sentence import Sentence
from database_creation.coreference import Coreference
from database_creation.np import Np
from database_creation.word import Word
# endregion

# region Display parameters
Database.set_parameters(to_print=['articles'],
                        print_attribute=True,
                        random_print=False,
                        limit_print=10)

Article.set_parameters(to_print=['entities', 'coreferences', 'tuple_contexts'],
                       print_attribute=True)

Coreference.set_parameters(to_print=['entity', 'representative', 'mentions'],
                           print_attribute=True)

Sentence.set_parameters(to_print=['text'],
                        print_attribute=False)

Np.set_parameters(to_print=['words'],
                  print_attribute=False)

Word.set_parameters(to_print=['text'],
                    print_attribute=False)
# endregion

database = Database(max_size=10000)
database.preprocess_tuples(limit=100, display=True)

database.process_tuples()

print(database)
