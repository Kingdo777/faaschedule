import os
import random

import pandas as pd
from multiprocessing import Pool
from LambdaData import *
import pickle
from math import ceil

from mypath import *


def load_trace(pth: str) -> list:
    if not os.path.exists(pth):
        raise FileNotFoundError(pth)

    with open(pth, "r+b") as f:
        # format: (lambdas, trace) =>
        # lambdas:    dict[func_name] = (mem_size, cold_time, warm_time)
        # trace = [LambdaData(func_name, mem, cold_time, warm_time), start_time]
        lambdas, trace = pickle.load(f)

    return trace


def test_FCFS(traces):
    n = len(traces)

    completion_times = [0] * n
    completion_times[0] = traces[0][0] + traces[0][1]
    current_time = completion_times[0]

    for i in range(1, len(traces)):
        run_time = traces[i][1]
        arrival_time = traces[i][0]
        start_time = max(current_time, arrival_time)

        completion_times[i] = start_time + run_time - arrival_time
        current_time = completion_times[i] + arrival_time

    # print(completion_times)

    # print("FCFS Average completion time:", sum(completion_times) / n)
    return completion_times


def test_SJF(traces):
    n = len(traces)

    import heapq
    completion_times = [0] * n
    ready_queue = []
    current_time = 0
    index = 0

    while index < n or ready_queue:
        while index < n and traces[index][0] <= current_time:
            heapq.heappush(ready_queue, (traces[index][1], index))
            index += 1

        if ready_queue:
            run_time, i = heapq.heappop(ready_queue)
            completion_times[i] = current_time + run_time - traces[i][0]
            current_time += run_time
        else:
            current_time = traces[index][0]

    # print(completion_times)
    # print("SJF Average completion time:", sum(completion_times) / n)
    return completion_times


def test_BJF(traces):
    n = len(traces)

    import heapq
    completion_times = [0] * n
    ready_queue = []
    current_time = 0
    index = 0

    while index < n or ready_queue:
        while index < n and traces[index][0] <= current_time:
            heapq.heappush(ready_queue, (traces[index][1] + traces[index][0], traces[index][1], index))
            index += 1

        if ready_queue:
            _, run_time, i = heapq.heappop(ready_queue)
            completion_times[i] = current_time + run_time - traces[i][0]
            current_time += run_time
        else:
            current_time = traces[index][0]

    # print(completion_times)
    # print("BJF Average completion time:", sum(completion_times) / n)
    return completion_times


def test_FaaSchedule(traces):
    n = len(traces)

    import heapq
    completion_times = [0] * n
    ready_queue = []
    current_time = 0
    update_time = 0
    index = 0

    while index < n or ready_queue:
        while index < n and traces[index][0] <= current_time:
            if ready_queue:
                for i in range(len(ready_queue)):
                    ready_queue[i][0] -= ((traces[index][0] - update_time) / len(ready_queue))
                update_time = traces[index][0]

            heapq.heappush(ready_queue, [
                traces[index][1], traces[index][1], index])
            index += 1

        if ready_queue:
            _, run_time, i = heapq.heappop(ready_queue)
            completion_times[i] = current_time + run_time - traces[i][0]
            current_time += run_time
        else:
            current_time = traces[index][0]

    # print(completion_times)
    # print("BJF Average completion time:", sum(completion_times) / n)
    return completion_times


def test_RR(traces, time_slice=1.0):
    n = len(traces)

    from collections import deque
    completion_times = [0] * n
    ready_queue = deque()
    current_time = 0
    index = 0

    while index < n or ready_queue:
        while index < n and traces[index][0] <= current_time:
            ready_queue.append((traces[index][1], index))
            index += 1

        if ready_queue:
            run_time, i = ready_queue.popleft()
            if run_time > time_slice:
                ready_queue.append((run_time - time_slice, i))
                completion_times[i] = current_time + time_slice
                current_time += time_slice
            else:
                completion_times[i] = current_time + run_time - traces[i][0]
                current_time += run_time
        else:
            current_time = traces[index][0]

    # print(completion_times)
    # print("RR Average completion time:", sum(completion_times) / n)

    return completion_times


def test_real_CFS(traces):
    n = len(traces)

    import heapq
    completion_times = [0] * n
    running_list = []
    index = 0

    arrive_time_point = traces[0][0]

    while index < n or running_list:
        running_list_len = len(running_list)

        old_arrive_time_point = arrive_time_point
        if index < n:
            arrive_time_point = traces[index][0]
            pass_time = arrive_time_point - old_arrive_time_point
        else:
            pass_time = -1

        while (pass_time > 0 or pass_time == -1) and running_list:
            run_time, start_time, i = heapq.heappop(running_list)
            # assert run_time != 0
            if pass_time > 0 and run_time > pass_time / running_list_len:
                heapq.heappush(running_list, (run_time, start_time, i))
                for j in range(running_list_len):
                    running_list[j] = (running_list[j][0] - pass_time / running_list_len,
                                       running_list[j][1],
                                       running_list[j][2])
                break
            else:
                completion_times[i] = old_arrive_time_point - start_time + run_time * running_list_len
                while True and running_list:
                    run_time_, start_time_, i_ = heapq.heappop(running_list)
                    if run_time_ == run_time:
                        completion_times[i_] = old_arrive_time_point - start_time_ + run_time * running_list_len
                    else:
                        heapq.heappush(running_list, (run_time_, start_time_, i_))
                        break
                for j in range(len(running_list)):
                    running_list[j] = (running_list[j][0] - run_time, running_list[j][1], running_list[j][2])
                if pass_time > 0:
                    pass_time -= run_time * running_list_len
                old_arrive_time_point += run_time * running_list_len
                running_list_len = len(running_list)

        while index < n and traces[index][0] == arrive_time_point:
            heapq.heappush(running_list, (traces[index][1], traces[index][0], index))
            index += 1

    print(completion_times)
    return completion_times


def get_fairness(l, l_f):
    aif = 1
    rif = 1
    aif_list = []
    rif_list = []
    for i in range(len(l)):
        if l[i] > l_f[i]:
            aif_list.append(l_f[i] / l[i])
            rif_list.append(l_f[i] / l[i])
        else:
            rif_list.append(1)

    if aif_list:
        aif = sum(aif_list) / len(aif_list)
    if rif_list:
        rif = sum(rif_list) / len(rif_list)

    return aif, rif


def main():
    traces = [
        (0, 2, 1),
        (1, 2, 1),
        (2, 2, 1),
        # (0, 2, 1),
        # (0, 2, 1),
    ]

    print("CFS Average completion time:", sum(test_real_CFS(traces)) / len(traces))


def main_():
    trace_path = os.path.join(RareSampleTracePath, "1000-a-ten-minutes.pckl")
    # trace_path = os.path.join(RandomSampleTracePath, "200-a-ten-minutes.pckl")
    # trace_path = os.path.join(RepresentativeSampleTracePath, "200-a-ten-minutes.pckl")
    traces_ = load_trace(trace_path)
    print("invocations:{}, Req/s:{}", len(traces_), len(traces_) / 600)

    traces = []
    for trace in traces_:
        traces.append((int(trace[1]) - NOON_MS, int(trace[0].warm_time), int(trace[0].run_time)))
    # print(len(traces))
    # traces = random.sample(traces, 100)
    # print(len(traces))
    traces.sort(key=lambda x: x[0])

    RR_completion_times = test_RR(traces, 1.0)
    FCFS_completion_times = test_FCFS(traces)
    SJF_completion_times = test_SJF(traces)
    BJF_completion_times = test_BJF(traces)
    FaaSchedule_completion_times = test_FaaSchedule(traces)

    print("RR Average completion time:", sum(RR_completion_times) / len(traces))
    print("FCFS Average completion time:", sum(FCFS_completion_times) / len(traces))
    print("SJF Average completion time:", sum(SJF_completion_times) / len(traces))
    print("BJF Average completion time:", sum(BJF_completion_times) / len(traces))
    print("FaaSchedule Average completion time:", sum(FaaSchedule_completion_times) / len(traces))

    print("RR fairness:", get_fairness(RR_completion_times, FaaSchedule_completion_times))
    print("FCFS fairness:", get_fairness(FCFS_completion_times, FaaSchedule_completion_times))
    print("SJF fairness:", get_fairness(SJF_completion_times, FaaSchedule_completion_times))
    print("BJF fairness:", get_fairness(BJF_completion_times, FaaSchedule_completion_times))
    print("FaaSchedule fairness:", get_fairness(FaaSchedule_completion_times, FaaSchedule_completion_times))


if __name__ == "__main__":
    main()
