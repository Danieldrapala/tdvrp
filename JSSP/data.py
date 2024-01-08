import re
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd


class Task:
    def __init__(self, job_id, task_id, sequence, usable_machines):
        self._job_id = job_id
        self._task_id = task_id
        self._sequence = sequence
        self._usable_machines = usable_machines

    def get_job_id(self):
        return self._job_id

    def get_task_id(self):
        return self._task_id

    def get_sequence(self):
        return self._sequence

    def get_usable_machines(self):
        return self._usable_machines

    def __eq__(self, other):
        return self._job_id == other.get_job_id() \
               and self._task_id == other.get_task_id() \
               and self._sequence == other.get_sequence() \
               and np.array_equal(self._usable_machines, other.get_usable_machines())

    def __str__(self):
        return f"[{self._job_id}, " \
            f"{self._task_id}, " \
            f"{self._sequence}, " \
            f"{self._usable_machines}, "

class Job:
    def __init__(self, job_id):
        self._job_id = job_id
        self._tasks = []
        self._max_sequence = 0

    def set_max_sequence(self, max_sequence):
        self._max_sequence = max_sequence

    def get_max_sequence(self):
        return self._max_sequence

    def get_tasks(self):
        return self._tasks

    def get_task(self, task_id):
        return self._tasks[task_id]

    def get_job_id(self):
        return self._job_id

    def get_number_of_tasks(self):
        return len(self._tasks)

    def __eq__(self, other):
        return self._job_id == other.get_job_id() \
               and self._max_sequence == other.get_max_sequence() \
               and self._tasks == other.get_tasks()


class Data(ABC):
    """
    Base class for JSSP instance data.
    """

    def __init__(self):

        self.sequence_dependency_matrix = None
        "2d nparray of sequence dependency matrix"

        self.job_task_index_matrix = None
        "2d nparray of (job, task): index mapping"

        self.usable_machines_matrix = None
        "2d nparray of usable machines"

        self.task_processing_times_matrix = None
        "2d nparray of task processing times on machines"

        self.machine_speeds = None
        "1d nparray of machine speeds"

        self.jobs = []
        "list of all Job instances"

        self.total_number_of_jobs = 0
        self.total_number_of_tasks = 0
        self.total_number_of_machines = 0
        self.max_tasks_for_a_job = 0

    def get_setup_time(self, job1_id, job1_task_id, job2_id, job2_task_id):
        if min(job1_id, job1_task_id, job2_id, job2_task_id) < 0:
            return 0

        return self.sequence_dependency_matrix[
            self.job_task_index_matrix[job1_id, job1_task_id],
            self.job_task_index_matrix[job2_id, job2_task_id]
        ]

    def get_runtime(self, job_id, task_id, machine):
        return self.task_processing_times_matrix[self.job_task_index_matrix[job_id, task_id], machine]

    def get_job(self, job_id):
        return self.jobs[job_id]

    def __str__(self):
        result = f"total jobs = {self.total_number_of_jobs}\n" \
                 f"total tasks = {self.total_number_of_tasks}\n" \
                 f"total machines = {self.total_number_of_machines}\n" \
                 f"max tasks for a job = {self.max_tasks_for_a_job}\n" \
                 f"tasks:\n" \
                 f"[jobId, taskId, sequence, usable_machines]\n"

        for job in self.jobs:
            for task in job.get_tasks():
                result += str(task) + '\n'

        if self.sequence_dependency_matrix is not None:
            result += f"sequence_dependency_matrix: {self.sequence_dependency_matrix.shape}\n\n" \
                      f"{self.sequence_dependency_matrix}\n\n"

        if self.job_task_index_matrix is not None:
            result += f"dependency_matrix_index_encoding: {self.job_task_index_matrix.shape}\n\n" \
                      f"{self.job_task_index_matrix}\n\n"

        if self.usable_machines_matrix is not None:
            result += f"usable_machines_matrix: {self.usable_machines_matrix.shape}\n\n" \
                      f"{self.usable_machines_matrix}\n\n"

        if self.task_processing_times_matrix is not None:
            result += f"task_processing_times: {self.task_processing_times_matrix.shape}\n\n" \
                      f"{self.task_processing_times_matrix}\n\n"

        if self.machine_speeds is not None:
            result += f"machine_speeds: {self.machine_speeds.shape}\n\n" \
                      f"{self.machine_speeds}"

        return result

class SpreadsheetData(Data):
    def __init__(self, seq_dep_matrix, machine_speeds, job_tasks):
        super().__init__()

        def _convert_to_df(path):
            path = Path(path)
            if path.suffix == ".csv":
                return pd.read_csv(path)
            elif path.suffix == ".xlsx":
                return pd.read_excel(path)
            else:
                raise UserWarning("File extension must either be .csv or .xlsx")

        if isinstance(job_tasks, pd.DataFrame):
            self.job_tasks_df = job_tasks
        else:
            self.job_tasks_df = _convert_to_df(job_tasks)

        if seq_dep_matrix is None or isinstance(seq_dep_matrix, pd.DataFrame):
            self.seq_dep_matrix_df = seq_dep_matrix
        else:
            self.seq_dep_matrix_df = _convert_to_df(seq_dep_matrix)

        if isinstance(machine_speeds, pd.DataFrame):
            self.machine_speeds_df = machine_speeds
        else:
            self.machine_speeds_df = _convert_to_df(machine_speeds)

        self._read_job_tasks_df(self.job_tasks_df)
        self._read_machine_speeds_df(self.machine_speeds_df)
        if self.seq_dep_matrix_df is not None:
            self._read_sequence_dependency_matrix_df(self.seq_dep_matrix_df)
        else:
            num_tasks = self.job_tasks_df.shape[0]
            self.sequence_dependency_matrix = np.zeros((num_tasks, num_tasks), dtype=np.intc)

        self.total_number_of_jobs = len(self.jobs)
        self.total_number_of_tasks = sum(len(job.get_tasks()) for job in self.jobs)
        self.max_tasks_for_a_job = max(job.get_number_of_tasks() for job in self.jobs)
        self.total_number_of_machines = self.machine_speeds.shape[0]

        self.job_task_index_matrix = np.full((self.total_number_of_jobs, self.max_tasks_for_a_job), -1, dtype=np.intc)
        self.usable_machines_matrix = np.empty((self.total_number_of_tasks, self.total_number_of_machines), dtype=np.intc)
        self.task_processing_times_matrix = np.full((self.total_number_of_tasks, self.total_number_of_machines), -1, dtype=np.float)

        # process all job-tasks
        task_index = 0
        for job in self.jobs:
            for task in job.get_tasks():

                # create mapping of (job id, task id) to index
                self.job_task_index_matrix[job.get_job_id(), task.get_task_id()] = task_index

                # create row in usable_machines_matrix
                self.usable_machines_matrix[task_index] = np.resize(task.get_usable_machines(),
                                                                    self.total_number_of_machines)

                # create row in task_processing_times
                for machine in task.get_usable_machines():
                    self.task_processing_times_matrix[task_index, machine] = self.machine_speeds[
                        machine]

                task_index += 1

    def _read_job_tasks_df(self, job_tasks_df):

        seen_jobs_ids = set()
        for i, row in job_tasks_df.iterrows():
            # create task object
            task = Task(
                int(row['Job']),
                int(row['Task']),
                int(row['Sequence']),
                np.array([int(x) for x in row['Usable_Machines'][1:-1].strip().split(' ')], dtype=np.intc)
            )
            job_id = task.get_job_id()

            # create & append new job if we encounter job_id that has not been seen
            if job_id not in seen_jobs_ids:
                self.jobs.append(Job(job_id))
                seen_jobs_ids.add(job_id)

            # update job's max sequence number
            if task.get_sequence() > self.get_job(job_id).get_max_sequence():
                self.get_job(job_id).set_max_sequence(task.get_sequence())

            # append task to associated job.tasks list
            self.get_job(job_id).get_tasks().append(task)

    def _read_sequence_dependency_matrix_df(self, seq_dep_matrix_df):
        seq_dep_matrix_df = seq_dep_matrix_df.drop(seq_dep_matrix_df.columns[0], axis=1)  # drop first column
        tmp = []
        for r, row in seq_dep_matrix_df.iterrows():
            tmp2 = []
            for c, value in row.iteritems():
                tmp2.append(value)
            tmp.append(tmp2)

        self.sequence_dependency_matrix = np.array(tmp, dtype=np.intc)

    def _read_machine_speeds_df(self, machine_speeds_df):
        machine_speeds_df = machine_speeds_df.sort_values('Machine')
        self.machine_speeds = np.array([row['RunSpeed'] for i, row in machine_speeds_df.iterrows()], dtype=np.float)
