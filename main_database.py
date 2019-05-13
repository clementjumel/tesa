from database_creation.database import Database

database = Database(year=2000, max_size=1000)

database.preprocess()
database.process()

database.write('out')
