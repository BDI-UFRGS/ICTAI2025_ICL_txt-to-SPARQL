from sparql_query import GoldSPARQLQuery, ExampleSPARQLQuery
from nl_query import NLQuery
from persistence_utils import load_object
from enum import Enum
import pickle
import json
import os
import uuid

class SupportedBenchmarks(Enum):
    LCQUAD = 'LCQUAD'

class Benchmark():

    def __init__(self, benchmark: SupportedBenchmarks):
        self.__benchmark = benchmark

    @property
    def benchmark(self):
        return self.__benchmark
    
    @benchmark.setter
    def benchmark(self, benchmark: SupportedBenchmarks):
        self.__benchmark = benchmark

    def load_gold_queries(self):
        if self.benchmark == SupportedBenchmarks.LCQUAD and not os.path.exists('../data/persistence/LCQUAD/gold_queries'):
            os.makedirs('../data/persistence/LCQUAD/gold_queries')
            with open('../data/benchmarks/LCQUAD/test.json', 'r') as file:
                for register in json.load(file):
                    gold_query = GoldSPARQLQuery(sparql_query=register['sparql_query'], nl_query=NLQuery(nl_query=register['corrected_question']))
                    with open(f'../data/persistence/LCQUAD/gold_queries/{gold_query.id}.pkl', 'wb') as dump_file:
                        pickle.dump(gold_query, dump_file)
                        yield gold_query
        elif self.benchmark == SupportedBenchmarks.LCQUAD and os.path.exists('../data/persistence/LCQUAD/gold_queries'):
            for file in os.listdir('../data/persistence/LCQUAD/gold_queries'):
                loaded_object = load_object(f'../data/persistence/LCQUAD/gold_queries/{file}')
                if not hasattr(loaded_object, "_id"):
                    loaded_object._id = file.replace('.pkl', '')
                yield loaded_object

    def load_example_queries(self):
        if self.benchmark == SupportedBenchmarks.LCQUAD and not os.path.exists('../data/persistence/LCQUAD/example_queries'):
            os.makedirs('../data/persistence/LCQUAD/example_queries')
            with open('../data/benchmarks/LCQUAD/train.json', 'r') as file:
                for register in json.load(file):
                    example_query = ExampleSPARQLQuery(sparql_query=register['sparql_query'], nl_query=NLQuery(nl_query=register['corrected_question']))
                    with open(f'../data/persistence/LCQUAD/example_queries/{example_query.id}.pkl', 'wb') as dump_file:
                        pickle.dump(example_query, dump_file)
                        yield example_query
        elif self.benchmark == SupportedBenchmarks.LCQUAD and os.path.exists('../data/persistence/LCQUAD/example_queries'):
            for file in os.listdir('../data/persistence/LCQUAD/example_queries'):
                loaded_object = load_object(f'../data/persistence/LCQUAD/example_queries/{file}')
                if not hasattr(loaded_object, "_id"):
                    loaded_object._id = file.replace('.pkl', '')
                yield loaded_object