from ..exception import InfeasibleSolutionException
import numpy as np

def compute_machine_makespans( operation_2d_array,task_processing_times_matrix,sequence_dependency_matrix,job_task_index_matrix):
    """
    Computes a 1d nparray of all the machine's makespan times given a 2d nparray of operations, where an operation
    is a 1d nparray of integers in the form [job_id, task_id, sequence, machine].

    :type operation_2d_array: nparray
    :param operation_2d_array: nparray of operations to compute the machine makespans for
    
    :type task_processing_times_matrix: nparray
    :param task_processing_times_matrix: task processing times matrix from static Data
    
    :type sequence_dependency_matrix: nparray
    :param sequence_dependency_matrix: sequence dependency matrix from static Data
    
    :type job_task_index_matrix: nparray
    :param job_task_index_matrix: job task index matrix from static Data
    
    :rtype: nparray
    :returns: memory view of a 1d nparray of machine make span times, where makespan[i] = makespan of machine i
    :raise: InfeasibleSolutionException if the solution is infeasible
    """
    num_jobs = sequence_dependency_matrix.shape[0]
    num_machines = task_processing_times_matrix.shape[1]
    machine_makespan_memory = np.zeros(num_machines)

    for row in range(operation_2d_array.shape[0]):
        job_id = operation_2d_array[row, 0]
        task_id = operation_2d_array[row, 1]
        sequence = operation_2d_array[row, 2]
        machine = operation_2d_array[row, 3]

        if machine_jobs_memory[machine] != -1:
            cur_task_index = job_task_index_matrix[job_id, task_id]
            prev_task_index = job_task_index_matrix[machine_jobs_memory[machine], machine_tasks_memory[machine]]
            setup = sequence_dependency_matrix[cur_task_index, prev_task_index]
        else:
            setup = 0

        if setup < 0 or sequence < job_seq_memory[job_id]:
            raise InfeasibleSolutionException()

        if job_seq_memory[job_id] < sequence:
            prev_job_end_memory[job_id] = job_end_memory[job_id]

        if prev_job_end_memory[job_id] <= machine_makespan_memory[machine]:
            wait = 0
        else:
            wait = prev_job_end_memory[job_id] - machine_makespan_memory[machine]

        runtime = task_processing_times_matrix[job_task_index_matrix[job_id, task_id], machine]

        # compute total added time and update memory modules
        machine_makespan_memory[machine] += runtime + wait + setup
        job_end_memory[job_id] = max(machine_makespan_memory[machine], job_end_memory[job_id])
        job_seq_memory[job_id] = sequence
        machine_jobs_memory[machine] = job_id
        machine_tasks_memory[machine] = task_id
    return machine_makespan_memory
