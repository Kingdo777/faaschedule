import os
import random

import pandas as pd
from multiprocessing import Pool
from LambdaData import *
import pickle
from math import ceil

from mypath import *

buckets = [str(i) for i in range(1, 1441)]
quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]


def trace_row(data):
    index, row = data
    secs_p_min = 60
    milis_p_sec = 1000
    trace = list()
    lambdas = dict()
    cold_dur = int(row["Maximum"])
    warm_dur = int(row["Average"])
    mem = int(row["divvied"])
    k = index
    d = LambdaData(k, mem, cold_dur, warm_dur)
    lambdas[k] = (k, mem, cold_dur, warm_dur)
    # 生成trace, 即每个function的特定请求发出的时间
    for minute, invocs in enumerate(row[buckets]):
        start = minute * secs_p_min * milis_p_sec
        if invocs == 0:
            continue
        elif invocs == 1:
            trace.append((d, start + random.uniform(0, secs_p_min * milis_p_sec)))
        else:
            every = (secs_p_min * milis_p_sec) / invocs
            trace += [(d, start + i * every + random.uniform(0, every)) for i in range(invocs)]
    out_trace = sorted(trace, key=lambda x: x[1])  # (lamdata, t)

    out_pth = os.path.join(FunctionsPath, index + ".pckl")
    # files are traces themselves. Just need to combine them together
    # format: (lambdas, trace) =>
    #   lambdas :   dict[func_name] = (mem_size, cold_time, warm_time)
    #   trace   :   [(LambdaData(func_name, mem, cold_time, warm_time), start_time)]
    with open(out_pth, "w+b") as f:
        pickle.dump((lambdas, out_trace), f)


def get_merged_trace() -> pd.DataFrame:
    dim_file = os.path.join(MergeTracePath, DurationsInvocationsMemoryFramePklFilename)
    merged_frame: pd.DataFrame
    if os.path.exists(dim_file):
        merged_frame = pickle.load(open(dim_file, "rb"))
    else:
        d_file = os.path.join(OriginalTracePath, DurationsFilename)
        durations = pd.read_csv(str(d_file))
        durations.index = durations["HashFunction"]  # 设置index
        durations = durations.drop_duplicates("HashFunction")  # 删除重复的行, 保留第一个重复项
        group_by_app = durations.groupby("HashApp").size()  # 按HashApp分组并获取每组的大小生成一个Series对象, index是HashApp

        i_file = os.path.join(OriginalTracePath, InvocationsFilename)
        invocations = pd.read_csv(str(i_file))
        invocations = invocations.dropna()  # dropna: 删除缺失值,即删除包含Nan的行
        invocations.index = invocations["HashFunction"]  # 设置index
        sums = invocations[buckets].sum(axis=1)  # sum: 按行求和, 生成的Series的index是invocations的index
        invocations = invocations[sums > 1]  # 选择sums>1的行
        invocations = invocations.drop_duplicates("HashFunction")  # 删除重复的行, 保留第一个重复项

        m_file = os.path.join(OriginalTracePath, MemoryFilename)
        memory = pd.read_csv(str(m_file))
        memory = memory.drop_duplicates("HashApp")
        memory.index = memory["HashApp"]

        # divive_by_func_num 用于计算每个函数的内存大小, 其值等于该函数所在应用程序的内存大小除以该应用程序的函数数量
        def divive_by_func_num(row):
            return ceil(row["AverageAllocatedMb"] / group_by_app[row["HashApp"]])

        # apply: 对DataFrame的每一行应用函数. axis=1表示按行应用, raw=False表示传递的函数接收的参数是Series对象,
        # result_type='expand'表示返回的结果是DataFrame对象
        new_mem = memory.apply(divive_by_func_num, axis=1, raw=False, result_type='expand')
        memory["divvied"] = new_mem

        # 合并DataFrame, 合并方式为内连接, 即只保留两者的索引(HashFunction)都存在的行
        # lsuffix, rsuffix: 如果两个DataFrame有相同的列名, 则会自动加后缀, lsuffix为左侧DataFrame的后缀, rsuffix为右侧DataFrame的后缀
        invocations_durations = invocations.join(durations, how="inner", lsuffix='', rsuffix='_durs')
        # on="HashApp": 以HashApp为键连接两个DataFrame, 即只保留两者的HashApp相同的行
        merged_frame = invocations_durations.join(memory, how="inner", on="HashApp",
                                                  lsuffix='', rsuffix='_mems')
        # 保存DataFrame
        if not os.path.exists(MergeTracePath):
            os.mkdir(MergeTracePath)
        with open(dim_file, "w+b") as f:
            pickle.dump(merged_frame, f)

    return merged_frame


def gen_trace(df_list, num_funcs, run, trace_path):
    lambdas = {}
    trace = []
    save_pth = os.path.join(trace_path, "{}-{}.pckl".format(num_funcs, run))
    if not os.path.exists(save_pth):
        for df in df_list:
            samp = df.sample(num_funcs // len(df_list))  # 从df中随机抽取num_funcs个样本
            for index, row in samp.iterrows():
                with open(os.path.join(FunctionsPath, "{}.pckl".format(index)), "r+b") as f:
                    one_lambda, one_trace = pickle.load(f)
                    lambdas = {**lambdas, **one_lambda}  # 合并字典, **表示解包，即将字典解包成关键字参数
                    trace += one_trace

        out_trace = sorted(trace, key=lambda x: x[1])  # 按照时间排序
        print(num_funcs, len(out_trace))
        with open(save_pth, "w+b") as f:
            data = (lambdas, out_trace)
            pickle.dump(data, f)

    print("done", save_pth)


def gen_traces(merged_trace: pd.DataFrame):
    # gen rare traces
    sums = merged_trace[buckets].sum(axis=1)  # 计算每一行的和
    qts = sums.quantile(quantiles)  # 计算分位数，即0.0, 0.25, 0.5, 0.75, 1.0分位数的值
    rare_df = merged_trace[sums.between(qts.iloc[0], qts.iloc[1])]
    gen_trace([rare_df], 1000, "a", RareSampleTracePath)

    # gen random traces
    gen_trace([merged_trace], 200, "a", RandomSampleTracePath)

    # gen representative traces
    sums = merged_trace["Average"]
    qts = sums.quantile(quantiles)  # 计算分位数，即0.0, 0.25, 0.5, 0.75, 1.0分位数的值
    bottom_qt_df = merged_trace[sums.between(qts.iloc[0], qts.iloc[1])]
    bottom_hlf_df = merged_trace[sums.between(qts.iloc[1], qts.iloc[2])]
    high_hlf_df = merged_trace[sums.between(qts.iloc[2], qts.iloc[3])]
    middle_df = merged_trace[sums.between(qts.iloc[3], qts.iloc[4])]
    gen_trace([bottom_qt_df, bottom_hlf_df, high_hlf_df, middle_df], 200, "a", RepresentativeSampleTracePath)


def convert_to_10_minute(filepath):
    noon_ms = NOON_MS
    noon_after = NOON_MS + 10 * 60 * 1000

    num_funcs, run = os.path.basename(filepath)[:-5].split("-")
    save_filename = "{}-{}-ten-minutes.pckl".format(num_funcs, run)
    save_pth = os.path.join(os.path.dirname(filepath), save_filename)

    if not os.path.exists(save_pth):
        with open(filepath, "r+b") as f:
            # format: (lambdas, trace) =>
            # lambdas:    dict[func_name] = (mem_size, cold_time, warm_time)
            # trace = [LambdaData(func_name, mem, cold_time, warm_time), start_time]
            lambdas, trace = pickle.load(f)

        start = stop = 0
        for i, (d, t) in enumerate(trace):
            if t > noon_ms:
                start = i
                break

        for i, (d, t) in enumerate(trace[start:]):
            if t > noon_after:
                stop = i + start
                break

        save_trc = trace[start:stop]
        print("form {} to {}".format(len(trace), len(save_trc)))
        with open(save_pth, "w+b") as f:
            # format: (lambdas, trace) =>
            # lambdas:    dict[func_name] = (mem_size, cold_time, warm_time)
            # trace = [LambdaData(func_name, mem, cold_time, warm_time), start_time]
            data = (lambdas, save_trc)
            pickle.dump(data, f)

    print("done", save_pth)


if __name__ == "__main__":
    merged_trace = get_merged_trace()

    functions = os.listdir(FunctionsPath)
    if len(functions) == 0:
        with Pool() as p:
            p.map(trace_row, merged_trace.iterrows())

    gen_traces(merged_trace)

    convert_to_10_minute(os.path.join(RareSampleTracePath, "1000-a.pckl"))
    convert_to_10_minute(os.path.join(RandomSampleTracePath, "200-a.pckl"))
    convert_to_10_minute(os.path.join(RepresentativeSampleTracePath, "200-a.pckl"))

    print("done")
