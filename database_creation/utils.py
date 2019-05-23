from copy import copy
from time import time
from re import findall


class BaseClass:
    # region Class base methods

    verbose = True
    to_print, print_attribute, print_lines, print_offsets = None, None, None, None

    def __str__(self):
        """
        Overrides the builtin str method for the instances of BaseClass.

        Returns:
            str, readable format of the instance.
        """

        to_print, print_attribute, print_lines, print_offsets = self.get_parameters()[:4]
        attributes = copy(to_print) or list(self.__dict__.keys())

        string = ''

        for attribute in attributes:
            s = self.to_string(getattr(self, attribute))
            string += self.prefix(print_attribute, print_lines, print_offsets, attribute) + s if s else ''

        return string

    @classmethod
    def get_parameters(cls):
        """
        Fetch the print attribute of the class.

        Returns:
            cls.to_print: list, attributes to print; if [], print all the attributes.
            cls.print_attribute: bool, whether or not to print the attributes' names.
            cls.print_lines: int, whether or not to print line breaks (and how many).
            cls.print_offsets: int, whether or not to print an offset (and how many).
        """

        return cls.to_print, cls.print_attribute, cls.print_lines, cls.print_offsets

    @classmethod
    def set_parameters(cls, to_print=None, print_attribute=None, print_lines=None, print_offsets=None):
        """
        Changes the print attributes of the class.

        Args:
            to_print: list, attributes to print; if [], print all the attributes.
            print_attribute: bool, whether or not to print the attributes' names.
            print_lines: int, whether or not to print line breaks (and how many).
            print_offsets: int, whether or not to print an offset (and how many).
        """

        cls.to_print = to_print if to_print is not None else cls.to_print
        cls.print_attribute = print_attribute if print_attribute is not None else cls.print_attribute
        cls.print_lines = print_lines if print_lines is not None else cls.print_lines
        cls.print_offsets = print_offsets if print_offsets is not None else cls.print_offsets

    @classmethod
    def set_verbose(cls, verbose):
        """
        Changes the verbose attribute of the class.

        Args:
            verbose: bool, new verbose value.
        """

        cls.verbose = verbose

    @staticmethod
    def to_string(item):
        """
        Converts an item of any type into a string with a easily readable format.

        Args:
            item: unk, item to convert into string, can be of any type.

        Returns:
            str, readable format of item.
        """

        if item is None or item == [] or item == () or item == {}:
            return ''

        elif isinstance(item, str):
            return item

        elif isinstance(item, (int, float)):
            return str(round(item, 2))

        elif isinstance(item, BaseClass):
            return str(item)

        elif isinstance(item, list):
            if isinstance(item[0], BaseClass):
                return ' '.join([str(ite) for ite in item if ite])
            elif isinstance(item[0], list):
                return '\n'.join([BaseClass.to_string(ite) for ite in item if ite])
            else:
                return ' | '.join([BaseClass.to_string(ite) for ite in item if ite])

        elif isinstance(item, set):
            return ' | '.join([BaseClass.to_string(ite) for ite in item if ite])

        elif isinstance(item, tuple):
            return ' '.join([BaseClass.to_string(ite) for ite in item if ite])

        elif isinstance(item, dict):
            if isinstance(list(item.keys())[0], int):
                return ' '.join([BaseClass.to_string(item[ite]) for ite in item])
            elif isinstance(list(item.keys())[0], (str, tuple)):
                if isinstance(item[list(item.keys())[0]], dict):
                    return '\n'.join([BaseClass.to_string(ite) + ':\n' +
                                      BaseClass.to_string(item[ite]) for ite in item])
                else:
                    return '\n'.join([BaseClass.to_string(ite) + ': ' + BaseClass.to_string(item[ite]) for ite in item])
        else:
            raise Exception("Unsupported type: {}.".format(type(item)))

    @staticmethod
    def prefix(print_attribute=False, print_lines=0, print_offsets=0, attribute=None):
        """
        Returns a prefix corresponding to the parameters.

        Args:
            print_attribute: bool, whether or not to print the attributes' names.
            print_lines: int, whether or not to print line breaks (and how many).
            print_offsets: int, whether or not to print an offset (and how many).
            attribute: str, attribute to print (if relevant).

        Returns:
            str, prefix corresponding to the parameters.
        """

        prefix = ''

        if print_lines:
            for _ in range(print_lines):
                prefix += '\n'

        if print_offsets:
            for _ in range(print_offsets):
                prefix += '  '

        if print_attribute:
            if attribute is not None:
                prefix += attribute + ': '
            else:
                raise Exception("No attribute specified.")

        return prefix

    # endregion

    # region Decorator

    class Verbose:
        """ Decorator for the display of a simple message. """

        def __init__(self, message):
            """ Initializes the Verbose decorator message. """

            self.message = message

        def __call__(self, func):
            """ Performs the call to the decorated function. """

            def f(*args, **kwargs):
                """ Decorated function. """

                slf = args[0]
                t0 = time()

                if slf.verbose:
                    print('\n' + self.message)

                func(*args, **kwargs)

                if slf.verbose:
                    print("Done (elapsed time: {}s).\n".format(round(time() - t0)))

            return f

    class Attribute:
        """ Decorator for monitoring an attribute. """

        def __init__(self, attribute):
            """ Initializes the Attribute decorator attribute. """

            self.attribute = attribute

        def __call__(self, func):
            """ Performs the call to the decorated function. """

            def f(*args, **kwargs):
                """ Decorated function. """

                slf = args[0]

                if slf.verbose:
                    print("Initial {}: {}".format(self.attribute, getattr(slf, self.attribute)))

                func(*args, **kwargs)

                if slf.verbose:
                    print("Final {}: {}".format(self.attribute, getattr(slf, self.attribute)))

            return f

    # endregion

    # region Methods standardize

    @staticmethod
    def standardize_location(entity):
        """
        Standardize a location entity (forget what is inside parenthesis).

        Args:
            entity: str, location entity to standardize.

        Returns:
            str, standardized entity.
        """

        before = findall(r'(.*?)\s*\(', entity)  # find the text before the parenthesis

        entity = before[0] if len(before) > 0 else entity

        return entity

    @staticmethod
    def standardize_person(entity):
        """
        Standardize a person entity (forget what is inside parenthesis, inverse last name and first name when necessary, 
        remove middle name/letter).

        Args:
            entity: str, person entity to standardize.

        Returns:
            str, standardized entity.
        """

        before = findall(r'(.*?)\s*\(', entity)  # find the text before the parenthesis

        entity = before[0] if len(before) > 0 else entity

        split = entity.split(', ')
        entity = ' '.join([split[1], split[0]]) if len(split) == 2 else entity

        split = entity.split()
        entity = ' '.join([split[0], split[2]]) if len(split) == 3 else entity

        return entity

    @staticmethod
    def standardize_organization(entity):
        """
        Standardize an organization entity (forget what is inside parenthesis).

        Args:
            entity: str, entity to standardize.

        Returns:
            str, standardized entity.
        """

        before = findall(r'(.*?)\s*\(', entity)  # find the text before the parenthesis

        entity = before[0] if len(before) > 0 else entity

        return entity

    @staticmethod
    def standardize(entity):
        """
        Standardize an entity by returning all possible different standardizations.

        Args:
            entity: str, entity to standardize.

        Returns:
            set, strings representing the different standardizations of the entity.
        """

        standardization = {entity}

        for standardize_name in ['location', 'person', 'organization']:
            standardize = getattr(BaseClass, 'standardize_' + standardize_name)
            s = standardize(entity)

            standardization.add(s)

            if standardize_name == 'person' and len(s.split()) == 2:
                standardization.add(s.split()[1])

        return standardization

    @staticmethod
    def match(entity1, entity2):
        """
        Check if the two entities match by checking the intersection of their standardization.

        Args:
            entity1: str, first entity to compare.
            entity2: str, second entity to compare.

        Returns:
            bool, True iff the entities match.
        """

        return True if BaseClass.standardize(entity1).intersection(BaseClass.standardize(entity2)) else False

    # endregion

    # region Other methods

    @staticmethod
    def subtuples(l):
        """
        Compute all the possible subtuples of len > 1 from sorted l. Note that the element inside a tuple will appear in
        the same order as in l.

        Args:
            l: list, original list.

        Returns:
            set, all the possible subtuples of len > 1 of l.
        """

        if len(l) == 2 or len(l) > 10:
            return {tuple(sorted(l))}

        else:
            res = {tuple(sorted(l))}
            for x in l:
                res = res.union(BaseClass.subtuples([y for y in l if y != x]))

            return res

    # endregion


def main():
    return


if __name__ == '__main__':
    main()
