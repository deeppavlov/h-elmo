x = input("Do you need the answer? (y/n): ")
if x.lower() == "y":
    required = True
else:
    required = False


def the_answer(self, *args):
    return 42


class EssentialAnswers(type):
    def __init__(cls, clsname, superclasses, attributedict):
        if required:
            cls.the_answer = the_answer


class Philosopher1(metaclass=EssentialAnswers):
    pass


class Philosopher2(metaclass=EssentialAnswers):
    pass


class Philosopher3(metaclass=EssentialAnswers):
    pass


plato = Philosopher1()
print(plato.the_answer())
kant = Philosopher2()
# let's see what Kant has to say :-)
print(kant.the_answer())