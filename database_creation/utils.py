from copy import copy


class BaseClass:
    to_print, print_attribute, print_lines, print_offsets = None, None, None, None

    def __str__(self):
        """
        Overrides the builtin str method for the instances of BaseClass.

        Returns:
            str, readable format of the instance.
        """

        to_print, print_attribute, print_lines, print_offsets = self.get_print_parameters()[:4]
        attributes = copy(to_print) or list(self.__dict__.keys())

        string = ''

        for attribute in attributes:
            s = self.to_string(getattr(self, attribute))
            string += self.prefix(print_attribute, print_lines, print_offsets, attribute) + s if s else ''

        return string

    @classmethod
    def set_print_parameters(cls, to_print=None, print_attribute=None, print_lines=None, print_offsets=None):
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
    def get_print_parameters(cls):
        """
        Computes the print attribute of the class.

        Returns:
            cls.to_print: list, attributes to print; if [], print all the attributes.
            cls.print_attribute: bool, whether or not to print the attributes' names.
            cls.print_lines: int, whether or not to print line breaks (and how many).
            cls.print_offsets: int, whether or not to print an offset (and how many).
        """

        return cls.to_print, cls.print_attribute, cls.print_lines, cls.print_offsets

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
                return ' '.join([str(ite) for ite in item if str(ite)])
            else:
                return '; '.join([str(ite) for ite in item if str(ite)])

        elif isinstance(item, tuple):
            return ' '.join([str(ite) for ite in item if str(ite)])

        elif isinstance(item, dict):
            return ''.join(['\n' + str(ite) + ': ' + str(item[ite]) for ite in item])

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


def main():
    return


if __name__ == '__main__':
    main()
