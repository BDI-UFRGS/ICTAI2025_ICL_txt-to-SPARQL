from enum import Enum
from sparql_query import GoldSPARQLQuery, ExampleSPARQLQuery
from typing import List

class SupportedTemplates(Enum):
    A = '../data/templates/A.txt'
    B = '../data/templates/B.txt'
    C = '../data/templates/B.txt'

def load_template(template: SupportedTemplates) -> str:
    with open(template.value, 'r') as file:
        return file.read()
    
def fill_template(template: SupportedTemplates, gold_query: GoldSPARQLQuery, example_queries: List[ExampleSPARQLQuery]) -> str:
    template = load_template(template)
    template = template.replace('<gold_query_nl>', gold_query.nl_query.nl_query)
    example_queries_str = ''
    for example_query in example_queries:
        example_queries_str += f'\nNL Query: {example_query.nl_query.nl_query}\nSparQL Query: {example_query.sparql_query}\n'
    template = template.replace('<example_queries>', example_queries_str)
    uris_str = ', '.join(gold_query.uris)
    template = template.replace('<useful_uris>', f'{uris_str}')
    return template