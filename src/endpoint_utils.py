from enum import Enum
from SPARQLWrapper import SPARQLWrapper, JSON
from sparql_query import SPARQLQuery, GeneratedSPARQLQuery, ConfusionMatrix

class SupportedEndpoints(Enum):
    DBPEDIA_2016_04 = 'http://localhost:8890/sparql'

class Endpoint:
    def __init__(self, endpoint: SupportedEndpoints):
        self._endpoint = endpoint

    def get_endpoint_output(self, query: SPARQLQuery):
        sparql = SPARQLWrapper(self._endpoint.value)
        sparql.setQuery(query.sparql_query)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(9000)
        
        try:
            results = sparql.query().convert()
        except Exception as e:
            print(f'error: {e}')
            return None
        
        if results == None:
            return None
        if query.query_type == 'COUNT':
            if results['results']['bindings']:
                return int(results['results']['bindings'][0]['count']['value'])
            else:
                return None
        if query.query_type == 'SELECT':
            if not results['results']['bindings']:
                return None
            elif results['results']['bindings'][0] == {}:
                return None
            return {item.get(list(item.keys())[0]).get('value') for item in results['results']['bindings']}
        if query.query_type == 'ASK':
            return results['boolean']
        
    def get_hallucinated_uris(self, query: GeneratedSPARQLQuery):
        
        generated_uris = query.uris
        hallucinated_uris = set()
        
        sparql = SPARQLWrapper(self._endpoint.value)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(9000)
        
        query_template = '''
            ASK {{
                {{
                    ?s ?p <{uri}> .
                }} UNION {{
                    <{uri}> ?p ?o .
                }} UNION {{
                   ?s <{uri}> ?o .
                }}
            }}
        '''
        
        if generated_uris == None:
            return None
        else:
            for uri in generated_uris:
                query = query_template.format(uri=uri)
                sparql.setQuery(query)
                if bool(sparql.query().convert()['boolean']) == False:
                    hallucinated_uris.add(uri)
            if len(hallucinated_uris) == 0:
                return None
            else:
                return hallucinated_uris
                
def fill_confusion_matrix(generated_query: GeneratedSPARQLQuery):
    
    if generated_query.query_type != generated_query.gold_query.query_type:
        generated_query.logs = 'The query types are different. The confusion matrix cannot be calculated.'
        return None
    
    if generated_query.has_query_parsing_error:
        generated_query.logs = 'The query has parsing errors. The confusion matrix cannot be calculated.'
        return None

    if generated_query.endpoint_output == None:
        generated_query.logs = 'The endpoint output is None. The confusion matrix cannot be calculated.'
        return None

    if generated_query.query_type == 'COUNT':

        if int(generated_query.endpoint_output) == int(generated_query.gold_query.endpoint_output):
            return ConfusionMatrix(tp=1, fp=0, fn=0)
        else:
            return ConfusionMatrix(tp=0, fp=1, fn=0)
        
    elif generated_query.query_type == 'ASK':
        if bool(generated_query.endpoint_output) == bool(generated_query.gold_query.endpoint_output):
            return ConfusionMatrix(tp=1, fp=0, fn=0)
        else:
            return ConfusionMatrix(tp=0, fp=1, fn=0)
        
    elif generated_query.query_type == 'SELECT' and generated_query.gold_query.endpoint_output == None:
        tp = 0
        fp = len(generated_query.endpoint_output)
        fn = 0
        
    elif generated_query.query_type == 'SELECT':
        tp = len(generated_query.gold_query.endpoint_output.intersection(generated_query.endpoint_output)) # tp (true positives): present in both the gold standard and the generated query responses.
        fp = len(generated_query.endpoint_output.difference(generated_query.gold_query.endpoint_output))   # fp (false positive): present in the generated query responses but not in the gold standard responses.
        fn = len(generated_query.gold_query.endpoint_output.difference(generated_query.endpoint_output))   # fn (false negative): present in the gold standard responses but not in the generated query responses. 
        return ConfusionMatrix(tp=tp, fp=fp, fn=fn)