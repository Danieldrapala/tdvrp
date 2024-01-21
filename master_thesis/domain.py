'''Basic domain class: Job, Machine and Operation.
'''


class IDInstance:
    def __init__(self, id: int) -> None:
        '''An instance with an ID.'''
        self.id = id

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.id})'


class Job(IDInstance):

    def __init__(self, id: int) -> None:
        '''Job instance.'''
        super().__init__(id)


class Machine(IDInstance):

    def __init__(self, id: int) -> None:
        '''Machine instance.
        '''
        super().__init__(id)


class Operation(IDInstance):

    def __init__(self, id: int, job: Job, machine: Machine, time: float) -> None:
        '''Operation instance.

        Args:
            id (int): Operation ID.
            job (Job): The job that this operation belonging to.
            machine (Machine): The machine that this operation assigned to.
            time (float): The processing time.
        '''
        super().__init__(id)

        # properties: keep constant
        self.__machine = machine
        self.__job = job
        self.__time = time

    @property
    def job(self): return self.__job

    @property
    def machine(self): return self.__machine

    @property
    def time(self): return self.__time


class Cloneable:
    def copy(self):
        class Empty(self.__class__):
            def __init__(self): pass

        cop = Empty()
        cop.__class__ = self.__class__
        cop.__dict__.update(self.__dict__)
        return cop
