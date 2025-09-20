from collections import defaultdict
from typing import List, Dict
from enum import Enum
from sklearn.metrics.pairwise import cosine_similarity
from sparql_query import ExampleSPARQLQuery, GoldSPARQLQuery
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
import timestamp_utils
import torch

class SupportedExampleSelections(Enum):
    EMBEDDING_SIMILARITY = "The embedding cosine similarity is the only selection criterion."
    RANDOMIC = "Random example selection"
    EMBEDDING_SIM_PLUS_DIVERSITY = "K-means clustering and cosine similarity to select examples from each cluster."

class ExampleSelection(ABC):

    def __init__(self, gold_query: GoldSPARQLQuery, selection_type: SupportedExampleSelections):
        self._selection_type = selection_type
        self._gold_query = gold_query
        self._selected_examples = []
        self._num_examples = 0
        self._logs = []

    @property
    def type(self):
        return self._selection_type
    
    @property
    def gold_query(self):
        return self._gold_query
    
    @property
    def selected_examples(self):
        return self._selected_examples

    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def logs(self):
        return self._logs

    def clear_examples(self):
        self._selected_examples = []
        self._num_examples = 0

    def remove_example(self, example: ExampleSPARQLQuery):
        for example_dict in self._selected_examples:
            if example_dict['example'] == example:
                self._selected_examples.remove(example_dict)
                self._reorder_and_rank_examples()
                self._num_examples = len(self._selected_examples)
                return

    @selected_examples.setter
    @abstractmethod
    def selected_examples(self, selected_examples: List[Dict]):
        pass

    @logs.setter
    def logs(self, message: str):
        self._logs.append(timestamp_utils.generate_log(message))

    @abstractmethod
    def _reorder_and_rank_examples(self):
        pass

class EmbeddingSimilaritySelection(ExampleSelection):

    def __init__(self, gold_query: GoldSPARQLQuery):
        super().__init__(gold_query, SupportedExampleSelections.EMBEDDING_SIMILARITY)
        self._selected_examples = self.selected_examples

    @ExampleSelection.selected_examples.setter
    def selected_examples(self, selected_examples: List[Dict]):
        for example_dict in selected_examples:
            self._selected_examples.append({
                'example': example_dict['example'],
                'cosine_similarity': example_dict['cosine_similarity'],
                'rank': None
            })
        self._reorder_and_rank_examples()
        self._num_examples = len(self._selected_examples)

    @abstractmethod
    def add_example(self, example: ExampleSPARQLQuery, *args, **kwargs):
        pass

    @abstractmethod
    def _select_examples(self, example_list: List[ExampleSPARQLQuery], num_examples: int, threshold: float = 0.0):
        pass

    def _reorder_and_rank_examples(self):
        self._selected_examples.sort(key=lambda ex: ex['cosine_similarity'], reverse=True)
        for index, example_dict in enumerate(self._selected_examples):
            example_dict['rank'] = index + 1

class BertSimilaritySelection(EmbeddingSimilaritySelection):

    def __init__(self, gold_query: GoldSPARQLQuery, example_list: List[ExampleSPARQLQuery], num_examples: int, threshold: float = 0.0):
        super().__init__(gold_query)
        self._select_examples(example_list, num_examples, threshold)
        self._num_examples = len(self._selected_examples)

    def _select_examples(self, example_list: List[ExampleSPARQLQuery], num_examples: int, threshold: float):
        self.clear_examples()

        temp_list = []

        for example_query in example_list:
            
            cosine_similarity = self._cosine_similarity(example_query, self.gold_query)
            
            if cosine_similarity > threshold:
                temp_list.append({
                    'example': example_query,
                    'cosine_similarity': cosine_similarity
                })

        temp_list.sort(key=lambda x: x['cosine_similarity'], reverse=True)

        for i in range(num_examples):
            try:
                self.add_example(temp_list[i]['example'])
            except:
                if len(temp_list) < (i+1):
                    self.logs = f"Warning: Not enough examples to select. Only {len(temp_list)} examples available, but {num_examples} requested."

    def add_example(self, example_query: ExampleSPARQLQuery):
        self._selected_examples.append({
            'example': example_query,
            'cosine_similarity': self._cosine_similarity(example_query, self.gold_query),
            'rank': None
        })
        self._reorder_and_rank_examples()
        self._num_examples += 1

    def _cosine_similarity(self, example_query: ExampleSPARQLQuery, gold_query: GoldSPARQLQuery):
        example_embedding = example_query.nl_query.get_bert_embedding
        gold_embedding = gold_query.nl_query.get_bert_embedding

        return cosine_similarity(example_embedding.detach().numpy(), gold_embedding.detach().numpy())[0][0]

class SimPlusDiversitySelection(ExampleSelection):
    def __init__(self, gold_query: GoldSPARQLQuery, example_list: List[ExampleSPARQLQuery], num_examples: int, threshold: float = 0.0):
        super().__init__(gold_query, SupportedExampleSelections.EMBEDDING_SIM_PLUS_DIVERSITY)
        self._select_examples(example_list, num_examples, threshold)
        self._num_examples = len(self._selected_examples)

    def _select_examples(self, example_list: List[ExampleSPARQLQuery], num_examples: int, threshold: float):
        self.clear_examples()
        temp_list = []

        embeddings = torch.stack([
            example_list[i].nl_query.embeddings[0]['embedding'].squeeze(0)
            for i in range(len(example_list))
        ])

        kmeans = KMeans(n_clusters=num_examples, random_state=42)
        kmeans.fit(embeddings.numpy())
        labels = kmeans.labels_

        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        for i in range(len(list(clusters.items()))):
            self.add_example((BertSimilaritySelection(gold_query=self.gold_query, example_list=[example_list[j] for j in list(clusters.items())[i][1]], num_examples=1).selected_examples)[0]['example'], int(list(clusters.items())[i][0]))

    def add_example(self, example_query: ExampleSPARQLQuery, cluster: int):
        self._selected_examples.append({
            'example': example_query,
            'cosine_similarity': self._cosine_similarity(example_query, self.gold_query),
            'cluster': cluster,
            'rank': None
        })
        self._reorder_and_rank_examples()
        self._num_examples += 1

    def _cosine_similarity(self, example_query: ExampleSPARQLQuery, gold_query: GoldSPARQLQuery):
        example_embedding = example_query.nl_query.get_bert_embedding
        gold_embedding = gold_query.nl_query.get_bert_embedding

        return cosine_similarity(example_embedding.detach().numpy(), gold_embedding.detach().numpy())[0][0]

    def _reorder_and_rank_examples(self):
        self._selected_examples.sort(key=lambda ex: ex['cosine_similarity'], reverse=True)
        for index, example_dict in enumerate(self._selected_examples):
            example_dict['rank'] = index + 1

    @ExampleSelection.selected_examples.setter
    def selected_examples(self, selected_examples: List[Dict]):
        for example_dict in selected_examples:
            self._selected_examples.append({
                'example': example_dict['example'],
                'cosine_similarity': example_dict['cosine_similarity'],
                'cluster': example_dict['cluster'],
                'rank': None
            })
        self._num_examples = len(self._selected_examples)

class RandomSelection(ExampleSelection):
    pass