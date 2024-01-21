

from .domain import (IDInstance, Operation)

class Step(IDInstance):
    
    def __init__(self, source:IDInstance=None) -> None:
        super().__init__(source.id if source else -1)
        self.__source = source

    @property
    def source(self): return self.__source

    @property
    def end_time(self) -> float:
        return 0.0


class JobStep(Step):
    
    def __init__(self, source:IDInstance=None) -> None:
        super().__init__(source=source)
        # pre-defined job chain
        self.__pre_job_op = None    # type: JobStep
        self.__next_job_op = None   # type: OperationStep
    
    @property
    def pre_job_op(self): return self.__pre_job_op

    @property
    def next_job_op(self): return self.__next_job_op

    @pre_job_op.setter
    def pre_job_op(self, op):
        self.__pre_job_op = op
        if hasattr(op, 'next_job_op'): op.__next_job_op = self

    
class MachineStep(Step):
    
    def __init__(self, source:IDInstance=None) -> None:
        super().__init__(source=source)
        self.__pre_machine_op = None
        self.__next_machine_op = None
    
    @property
    def pre_machine_op(self): return self.__pre_machine_op

    @property
    def next_machine_op(self): return self.__next_machine_op

    @pre_machine_op.setter
    def pre_machine_op(self, op):
        self.__pre_machine_op = op
        if hasattr(op, 'next_machine_op'): op.__next_machine_op = self
    
    @property
    def tailed_machine_op(self):
        step = self
        while True:
            if step.__next_machine_op is None: return step
            step = step.__next_machine_op

class OperationStep(JobStep, MachineStep):

    def __init__(self, op: Operation=None) -> None:
        JobStep.__init__(self, op)
        MachineStep.__init__(self, op)

        # final variable in mathematical model, while shadow variable in disjunctive graph
        # model, i.e. the start time is determined by operation sequence
        self.__start_time = 0.0


    @property
    def start_time(self) -> float: return self.__start_time

    @property
    def end_time(self) -> float: return self.__start_time + self.source.time

    @property
    def tail(self) -> float:
        num = 0
        ref = self
        while ref:
            num += ref.source.time
            ref = ref.next_job_op
        return num

    def update_start_time(self, start_time:float=None):
        if start_time is not None:
            self.__start_time = start_time
        elif self.pre_machine_op:        
            self.__start_time = max(self.pre_job_op.end_time, self.pre_machine_op.end_time)


