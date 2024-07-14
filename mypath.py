import os.path as path
import platform

HomePath = "D:\\Documents\\PycharmProjects\\faaschedule"

NOON_MS = 12 * 60 * 60 * 1000

if platform.system() == "Windows":
    OriginalTracePath = path.join(HomePath, "data\\azure_trace\\original")
    MergeTracePath = path.join(HomePath, "data\\azure_trace\\merge")
    FunctionsPath = path.join(HomePath, "data\\azure_trace\\functions")
    SampleTracePath = path.join(HomePath, "data\\azure_trace\\sample-functions")
    OutputPath = path.join(HomePath, "data\\output")
    LogPath = path.join(HomePath, "data\\log")
else:
    OriginalTracePath = path.join(HomePath, "data/azure_trace/original")
    MergeTracePath = path.join(HomePath, "data/azure_trace/merge")
    FunctionsPath = path.join(HomePath, "data/azure_trace/functions")
    SampleTracePath = path.join(HomePath, "data/azure_trace/sample-functions")
    OutputPath = path.join(HomePath, "data/output")
    LogPath = path.join(HomePath, "data/log")

RareSampleTracePath = path.join(SampleTracePath, "rare")
RadeLargeSampleTracePath = path.join(SampleTracePath, "rare_large")
MemorySampleTracePath = path.join(SampleTracePath, "memory")
RandomSampleTracePath = path.join(SampleTracePath, "random")
RepresentativeSampleTracePath = path.join(SampleTracePath, "representative")

DurationsFilename = "function_durations_percentiles.anon.d01.csv"
InvocationsFilename = "invocations_per_function_md.anon.d01.csv"
MemoryFilename = "app_memory_percentiles.anon.d01.csv"
DurationsInvocationsMemoryFramePklFilename = "duration_invocations_memory_frame.pkl"

import os

if not os.path.exists(OriginalTracePath):
    os.makedirs(OriginalTracePath)
if not os.path.exists(MergeTracePath):
    os.makedirs(MergeTracePath)
if not os.path.exists(FunctionsPath):
    os.makedirs(FunctionsPath)
if not os.path.exists(SampleTracePath):
    os.makedirs(SampleTracePath)
if not os.path.exists(RareSampleTracePath):
    os.makedirs(RareSampleTracePath)
if not os.path.exists(RadeLargeSampleTracePath):
    os.makedirs(RadeLargeSampleTracePath)
if not os.path.exists(MemorySampleTracePath):
    os.makedirs(MemorySampleTracePath)
if not os.path.exists(RandomSampleTracePath):
    os.makedirs(RandomSampleTracePath)
if not os.path.exists(RepresentativeSampleTracePath):
    os.makedirs(RepresentativeSampleTracePath)
if not os.path.exists(OutputPath):
    os.makedirs(OutputPath)
if not os.path.exists(LogPath):
    os.makedirs(LogPath)
