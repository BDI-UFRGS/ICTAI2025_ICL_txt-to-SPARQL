from typing import List, Dict
from sparql_query import GoldSPARQLQuery, GeneratedSPARQLQuery, ExampleSPARQLQuery, ConfusionMatrix
from example_selection import SupportedExampleSelections, BertSimilaritySelection, SimPlusDiversitySelection
from nl_query import SupportedEmbedders
from llm_connector import SupportedModels, LLMApi
from decimal import Decimal
from prompting_utils import SupportedTemplates, fill_template
from timestamp_utils import generate_note, extract_timestamp_of
from tqdm import tqdm
from prompting_utils import fill_template
import datetime
import uuid

class Round: pass
class Execution: pass
class Experiment: pass

class Experiment:
    def __init__(self, description: str, prompt_template: SupportedTemplates, model: SupportedModels, rounds: List[Round] = None):
        self._id = str(uuid.uuid4())
        self._description = description
        self._prompt_template = prompt_template
        self._model = model
        self._rounds = rounds
        self._clusters = []
        self._notes = []

    @property
    def id(self):
        return self._id

    @property
    def description(self):
        return self._description
    
    @property
    def prompt_template(self):
        return self._prompt_template
    
    @property
    def model(self):
        return self._model
    
    @property
    def n_rounds(self):
        return len(self._rounds)
    
    @property
    def notes(self):
        return self._notes
    
    @property
    def rounds(self):
        return self._rounds
    
    @description.setter
    def description(self, description: str):
        self._description = description

    @prompt_template.setter
    def prompt_template(self, prompt_template: str):
        self._prompt_template = prompt_template

    @model.setter
    def model(self, model: SupportedModels):
        self._model = model.value

    @rounds.setter
    def rounds(self, rounds: List[Round]):
        self._rounds = rounds
        self._n_rounds = len(rounds)

    def add_round(self, round: Round):
        self._rounds.append(round)
        self._n_rounds += 1

    def add_note(self, note: str):
        self._notes.append(generate_note(note))

    def delete_note(self, timestamp: str):
        for note in self._notes:
            if extract_timestamp_of(note) == timestamp:
                self._notes.remove(note)

class Round:
    def __init__(self, experiment: Experiment, round_number: int, executions: List[Execution] = None):
        self._round_number = round_number
        self._executions = executions
        self._experiment = experiment
        self._notes = []

    @property
    def round_number(self):
        return self._round_number
    
    @property
    def executions(self):
        return self._executions
    
    @property
    def experiment(self):
        return self._experiment
    
    @property
    def macro_f1(self):
        return sum(
            (execution.generated_query.confusion_matrix.f1_score if execution.generated_query.confusion_matrix else 0)
            for execution in self._executions
        ) / len(self._executions)

    @property
    def micro_f1(self):
        tp = sum(execution.generated_query.confusion_matrix.tp for execution in self._executions)
        tn = sum(execution.generated_query.confusion_matrix.tn for execution in self._executions)
        fp = sum(execution.generated_query.confusion_matrix.fp for execution in self._executions)
        fn = sum(execution.generated_query.confusion_matrix.fn for execution in self._executions)

        return ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn).f1_score
          
    @property
    def semantic_accuracy(self):
        return sum(execution.generated_query.is_equivalent for execution in self._executions) / len(self._executions)
    
    @property
    def notes(self):
        return self._notes
    
    @executions.setter
    def executions(self, executions: List[Execution]):
        self._executions = executions

    @semantic_accuracy.setter
    def semantic_accuracy(self, semantic_accuracy: float):
        self._semantic_accuracy = semantic_accuracy

    def add_execution(self, execution):
        self._executions.append(execution)

    def add_note(self, note: str):
        self._notes.append(generate_note(note))

    def delete_note(self, timestamp: str):
        for note in self._notes:
            if extract_timestamp_of(note) == timestamp:
                self._notes.remove(note)

class Execution:
    def __init__(self, experiment: Experiment, gold_query: GoldSPARQLQuery, example_queries: List[Dict], exp_ts: datetime.datetime = None, sec_api_latency: Decimal = None):
        self._experiment = experiment
        self._gold_query = gold_query
        self._example_queries = example_queries
        self._exp_ts = exp_ts # This is the llm execution timestamp
        self._sec_api_latency = sec_api_latency
        self._generated_query = None
        self._notes = []

    def __str__(self):
        return (
            f"Execution(\n"
            f"  Experiment: {self._experiment},\n"
            f"  Gold Query: {self._gold_query},\n"
            f"  Example Queries: {self._example_queries},\n"
            f"  Timestamp: {self._exp_ts},\n"
            f"  API Latency: {self._sec_api_latency},\n"
            f"  Generated Query: {self._generated_query},\n"
            f"  Notes: {self._notes}\n"
            f")"
        )

    @property
    def experiment(self):
        return self._experiment

    @property
    def prompt(self):
        if self._gold_query != None and self._example_queries != None:
            return fill_template(self._experiment.prompt_template, self._gold_query, [item['example'] for item in self._example_queries])
        else:
            return None
    
    @property
    def generated_query(self):
        return self._generated_query
    
    @property
    def gold_query(self):
        return self._gold_query
    
    @property
    def example_queries(self):
        return self._example_queries
    
    @property
    def exp_ts(self):
        return self._exp_ts
    
    @property
    def sec_api_latency(self):
        return self._sec_api_latency
    
    @property
    def notes(self):
        return self._notes
    
    @generated_query.setter
    def generated_query(self, generated_query: GeneratedSPARQLQuery):
        self._generated_query = generated_query    
    
    @example_queries.setter
    def example_queries(self, example_queries: List[ExampleSPARQLQuery]):
        self._example_queries = example_queries

    @exp_ts.setter
    def exp_ts(self, exp_ts: datetime.datetime):
        self._exp_ts = exp_ts

    @sec_api_latency.setter
    def sec_api_latency(self, sec_api_latency: Decimal):
        self._sec_api_latency = sec_api_latency

    def add_note(self, note: str):
        self._notes.append(generate_note(note))

    def delete_note(self, timestamp: str):
        for note in self._notes:
            if extract_timestamp_of(note) == timestamp:
                self._notes.remove(note)

def start_experiment(description: str, model: SupportedModels, prompt_template: SupportedTemplates,selection_type: SupportedExampleSelections, embedder: SupportedEmbedders, n_rounds: int, n_examples: int, gold_queries: List[GoldSPARQLQuery], example_list: List[ExampleSPARQLQuery]):
    experiment = Experiment(description=description, prompt_template=prompt_template, model=model)
    rounds = []
    execution_list = []
    for gold_query in tqdm(gold_queries, desc=f"Selecting example queries and writing the prompt for each gold query", leave=False):
        if selection_type == SupportedExampleSelections.EMBEDDING_SIMILARITY and embedder == SupportedEmbedders.BERT:
            selected_examples = BertSimilaritySelection(gold_query=gold_query, example_list=example_list, num_examples=n_examples).selected_examples
        if selection_type == SupportedExampleSelections.EMBEDDING_SIM_PLUS_DIVERSITY and embedder == SupportedEmbedders.BERT:
            selected_examples = SimPlusDiversitySelection(gold_query=gold_query,example_list=example_list,num_examples=n_examples).selected_examples        
        execution = Execution(experiment = experiment, gold_query=gold_query, example_queries=selected_examples)
        execution_list.append(execution)
    for round_number in range(n_rounds):
        execution_list_copy = [
            Execution(
                experiment = execution.experiment,
                gold_query = execution.gold_query,
                example_queries = execution.example_queries
            ) for execution in execution_list
        ] 
        rounds.append(Round(experiment=experiment, round_number=round_number, executions=execution_list_copy))
    experiment.rounds = rounds
    return experiment