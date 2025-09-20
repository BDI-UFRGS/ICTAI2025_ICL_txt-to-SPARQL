from sklearn.metrics.pairwise import cosine_similarity
from rdflib.plugins.sparql import prepareQuery
from rdflib import Variable, URIRef
from itertools import permutations
from abc import abstractmethod
from llm_connector import SupportedModels
from nl_query import NLQuery
import timestamp_utils
import datetime
import pickle
import uuid
import re

class ConfusionMatrix:
    def __init__(self, tp: int, fp: int, fn: int, tn: int = 0):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn

    def __repr__(self):
        return f"ConfusionMatrix(tp={self.tp}, fp={self.fp}, fn={self.fn}, tn={self.tn})"
    
    @property
    def accuracy(self):
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total > 0 else 0

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0

    @property
    def f1_score(self):
        prec = self.precision
        rec = self.recall
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    @property
    def mcc(self):
        num = (self.tp * self.tn) - (self.fp * self.fn)
        den = ((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)) ** 0.5
        return num / den if den > 0 else 0
    
    @property
    def specificity(self):
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0
    
    @property
    def npv(self): # Negative Predictive Value
        return self.tn / (self.tn + self.fn) if (self.tn + self.fn) > 0 else 0

    @property
    def fp_rate(self):
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0
    
    @property
    def fd_rate(self): # False Discovery Rate
        return self.fp / (self.fp + self.tp) if (self.fp + self.tp) > 0 else 0
    
    @property
    def fn_rate(self):
        return self.fn / (self.fn + self.tp) if (self.fn + self.tp) > 0 else 0

class SPARQLQuery:

    def __init__(self, sparql_query: str, endpoint_output = None):

        self._id = uuid.uuid4()
        self._sparql_query = self._standardize_query_to_parser(sparql_query)
        self._has_filter = self.__has_filter()
        self._filter = self.__extract_filter(sparql_query) if self._has_filter else None
        self._formatted_triples = self.__extract_formatted_triples()
        self._triple_permutations = self.__generate_sparql_triple_permutations()
        self._endpoint_output = endpoint_output
        self._logs = []

    @property
    def id(self):
        return self._id

    @property
    def has_filter(self):
        return self._has_filter
    
    @property
    def sparql_query(self):
        return self._sparql_query
        
    @property
    def query_type(self):
        return self._get_sparql_query_type(self._sparql_query)
    
    @property
    def filter(self):
        return self._filter
    
    @property
    def triples(self):
        return self.__extract_triples(self.sparql_query)
    
    @property
    def variables(self):
        return self.__extract_variables()
    
    @property
    def uris(self):
        return self._extract_uris()
    
    @property
    def formatted_triples(self):
        return self._formatted_triples
    
    @property
    def triple_permutations(self):
        return self._triple_permutations
    
    @property
    def endpoint_output(self):
            return self._endpoint_output
    
    @property
    def has_query_parsing_error(self):
        try:
            prepareQuery(self._sparql_query)
            return False
        except Exception as e:
            self.logs = f"Parsing error: {e}"
            return True
    
    @property
    def logs(self):
        return self._logs

    @sparql_query.setter
    def sparql_query(self, sparql_query: str):
        self._sparql_query = self.standardize_query_to_parser(sparql_query)
        self._has_filter = self._has_filter(sparql_query)
        self._uris = self.extract_uris()
        self._formatted_triples = self.extract_formatted_triples()
        self._triple_permutations = self.generate_sparql_triple_permutations()

    @endpoint_output.setter
    def endpoint_output(self, endpoint_output):
        self._endpoint_output = endpoint_output

    @logs.setter
    def logs(self, message: str):
        self._logs.append(timestamp_utils.generate_log(message))

    def _flip_triple(self, triple : list):
        
        subj, pred, obj = triple
        
        if isinstance(subj, Variable): subj = f'?{subj}'
        elif isinstance(subj, str): subj = f'<{subj}>'
        
        if isinstance(pred, Variable): pred = f'?{pred}'
        elif isinstance(pred, str): pred = f'<{pred}>'
        
        if isinstance(obj, Variable): obj = f'?{obj}'
        elif isinstance(obj, str): obj = f'<{obj}>'
        
        return self._sparql_query.replace(f'{subj} {pred} {obj}', f'{obj} {pred} {subj}')

    def __replace_prefixes(self, query: str):
        prefixes = dict(re.findall(r'PREFIX (\w+): <([^>]+)>', query))
        
        if not prefixes:
            return query

        def replacing(match):
            prefix = match.group(1)
            return prefixes.get(prefix, prefix)
        
        query = re.sub(r'\b(\w+):', replacing, query)

        query = re.search(r'\b(ASK|SELECT|DESCRIBE|CONSTRUCT)\b.*\}', query, re.DOTALL)

        if query:
            return query.group(0)
        else:
            return None

    def __clean_after_last_brace(self, query: str):
        closing_braces = [m.start() for m in re.finditer(r'\}', query)]
        if not closing_braces:
            return query
        last_brace_index = closing_braces[-1]
        cleaned_query = query[:last_brace_index + 1]
        return cleaned_query

    def _standardize_query_to_parser(self, query):
        if isinstance(query, SPARQLQuery):
            query = query.sparql_query
        elif not isinstance(query, str):
            raise ValueError("The input must be a string. Received: ", type(query))
        query = re.sub(r'\n\s*', ' ', query) # Remove line breaks
        query = re.sub(r'\s+', ' ', query) # Remove multiple spaces
        query = re.sub(r'SELECT COUNT\((.*?)\)', r'SELECT (COUNT(\1) AS ?count)', query)
        query = re.sub(r'SELECT DISTINCT COUNT\((.*?)\)', r'SELECT DISTINCT (COUNT(\1) AS ?count)', query)
        query = re.sub(r'(<[^>]*?)\s+([^>]*>)', r'\1\2', query) # Remove blank spaces between URIs
        query = re.sub(r'dbpedia-owl:([^\s]+)', r'<http://dbpedia.org/ontology/\1>', query) # Change prefix to the complete uri reference
        query = re.sub(r"FILTER \(IS INSTANCE OF \((\?\w+), <([^>]+)>\)\)", r"\1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <\2>.", query)
        query = self.__replace_prefixes(query)
        query = re.sub(r"^.*?(ASK|SELECT|DESCRIBE|CONSTRUCT)", r"\1", query) # Remove any content before the query itself
        query = self.__clean_after_last_brace(query) # Remove any content after the query itself
        return query

    def __has_filter(self):
        return re.search(r"\bFILTER\s*\(", self._sparql_query, re.IGNORECASE) is not None

    def __extract_filter(self, query: str):
        return re.search(r'FILTER\s*\([^\)]+\)', query).group(0) if re.search(r'FILTER\s*\([^\)]+\)', query) else None
    
    def __extract_triples(self, query: str):
        
        try:
            prepared_query = prepareQuery(query)
        except:
            return None
        
        try:
            triples = prepared_query.algebra.p.p.p.p.p.p.p.triples
            if triples is not None:
                return triples
        except:
            try:
                triples = prepared_query.algebra.p.p.p.p.p.p.triples
                if triples is not None:
                    return triples
            except:
                try:
                    triples = prepared_query.algebra.p.p.p.p.p.triples
                    if triples is not None:
                        return triples
                except:
                    try:
                        triples = prepared_query.algebra.p.p.p.p.p1.triples
                        if triples is not None:
                            return triples
                    except:
                        try: 
                            triples = prepared_query.algebra.p.p.p.p.triples
                            if triples is not None:
                                return triples
                        except:
                            try:
                                triples = prepared_query.algebra.p.p.p.p1.p1.p1.triples
                                if triples is not None:
                                    return triples
                            except:
                                try:
                                    triples = prepared_query.algebra.p.p.p.p1.p1.triples
                                    if triples is not None:
                                        return triples
                                except:
                                    try:
                                        triples = prepared_query.algebra.p.p.p.p1.triples
                                        if triples is not None:
                                            return triples
                                    except:
                                        try:
                                            triples = prepared_query.algebra.p.p.p.triples
                                            if triples is not None:
                                                return triples
                                        except:
                                            triples = prepared_query.algebra.p.p.triples
                                            if triples is not None:
                                                return triples
                                            else:
                                                return "It was not possible to extract the triples from the query, despite the query being valid."
                                            
    def __extract_variables(self):
        try: 
            prepareQuery(self._sparql_query)
        except:
            return None
        
        if self.triples is None:
            return None

        variables = set()
        triples = self.triples
        for triple in triples:
            for element in triple:
                if isinstance(element, Variable):
                    variables.add(str(element))
        return variables
    
    def __extract_formatted_triples(self):
        try:
            prepareQuery(self._sparql_query)
        except:
            return None
        
        if self.triples is None:
            return None

        extracted_triples = list(reversed(list(self.triples)))
        formatted_tripples = []
        for s, p, o in extracted_triples:
            s_temp = f'?{s}' if isinstance(s, Variable) else f'<{s}>'
            p_temp = f'?{p}' if isinstance(p, Variable) else f'<{p}>'
            o_temp = f'?{o}' if isinstance(o, Variable) else f'<{o}>'
            formatted_tripples.append(f'{s_temp} {p_temp} {o_temp}')
        return formatted_tripples

    def _extract_uris(self):
        try: 
            prepareQuery(self._sparql_query)
        except:
            return None
        
        if self.triples is None:
            return None

        uris = set()
        for triple in self.triples:
            for element in triple:
                if isinstance(element, URIRef):
                    uris.add(str(element))
        return uris
    
    def __generate_sparql_triple_permutations(self):
        try: 
            prepareQuery(self._sparql_query)
        except:
            return None
        
        if self.triples is None:
            return None

        pre_where_part = (re.search(r'^(.*?)(?=\bWHERE\b)', self._sparql_query, re.IGNORECASE | re.DOTALL)).group(0).strip()
        triples = self._formatted_triples
        triple_permutations = permutations(triples)
        permuted_queries = []
        for perm in triple_permutations:
            new_where = ' .\n'.join(perm)
            filter = self._filter if self._has_filter else ''
            new_query = f"{pre_where_part} \nWHERE {{\n{new_where} \n{filter}\n}}" if self._has_filter else f"{pre_where_part} \nWHERE {{{new_where}\n}}"
            permuted_queries.append(self._standardize_query_to_parser(new_query))
        return permuted_queries

    def _get_sparql_query_type(self, query):

        if isinstance(query, SPARQLQuery):
            query = query.sparql_query
        elif not isinstance(query, str):
            raise ValueError("The input must be a string. Received: ", type(query))
        
        match_count = re.search(r".*\((COUNT\(.*?\) AS \?count\)).*", query, re.IGNORECASE)
        match = re.search(r"\b(ASK|SELECT)\b", re.sub(r"#.*", "", query), re.IGNORECASE)
        
        if match_count:
            return 'COUNT'
        if match:
            return match.group(1).upper()
        else:
            return None
        
class GoldSPARQLQuery(SPARQLQuery):
    
    def __init__(self, sparql_query: str, nl_query: NLQuery, endpoint_output = None):
        super().__init__(sparql_query, endpoint_output)
        self._f1_score = 1.0
        if isinstance(nl_query, NLQuery):
            self._nl_query = nl_query
        else:
            raise ValueError("The input must be a NLQuery object. Received: ", type(nl_query))
        self._identified_entities = None
    
    @property
    def nl_query(self) -> NLQuery:
        return self._nl_query
    
    @nl_query.setter
    def nl_query(self, nl_query: NLQuery):
        self._nl_query = nl_query

    def cosine_similarity_with(self, example_query):
        cos_sim = None
        if not isinstance(example_query, ExampleSPARQLQuery):
            ValueError("The input must be an ExampleSPARQLQuery object. Received: ", type(example_query))
        embedding1 = self._nl_query.embeddings[0]['embedding']
        embedding2 = example_query.nl_query.embeddings[0]['embedding']
        cos_sim = cosine_similarity(embedding1.detach().numpy(), embedding2.detach().numpy())[0][0]
        return cos_sim
    
    def add_identified_entities(model: SupportedModels):
        pass
        
class ExampleSPARQLQuery(SPARQLQuery):

    def __init__(self, sparql_query: str, nl_query: NLQuery, endpoint_output = None):
        super().__init__(sparql_query, endpoint_output)
        self._f1_score = 1.0
        if isinstance(nl_query, NLQuery):
            self._nl_query = nl_query
        else:
            raise ValueError("The input must be a NLQuery object. Received: ", type(nl_query))
        self._identified_entities = None

    @property
    def nl_query(self):
        return self._nl_query
    
    @nl_query.setter
    def nl_query(self, nl_query: str):
        self._nl_query = nl_query

    def cosine_similarity_with(self, gold_query: GoldSPARQLQuery):
        cos_sim = None
        embedding1 = self._nl_query.embeddings[0]['embedding']
        embedding2 = gold_query.nl_query.embeddings[0]['embedding']
        cos_sim = cosine_similarity(embedding1.detach().numpy(), embedding2.detach().numpy())[0][0]
        return cos_sim
    
    def add_identified_entities(model: SupportedModels):
        pass
        
class GeneratedSPARQLQuery(SPARQLQuery):
    
    def __init__(self, sparql_query: str, gold_query: GoldSPARQLQuery, endpoint_output = None):
        super().__init__(sparql_query, endpoint_output)
        self.__gold_query = gold_query
        if type(gold_query) is not GoldSPARQLQuery:
            raise ValueError("The input must be a GoldSPARQLQuery object. Received: ", type(gold_query))
        self._confusion_matrix = None
        self._hallucinated_uris = None

    @property
    def gold_query(self):
        return self.__gold_query
    
    @gold_query.setter
    def gold_query(self, gold_query: GoldSPARQLQuery):
        self.__gold_query = gold_query

    @property
    def nl_query(self):
        return self.__gold_query.nl_query
    
    @property
    def has_triple_flip(self):
        return self._has_triple_flip()
    
    @property
    def is_equivalent(self):
        try:
            if self.variables == None:
                self.logs = "No variables in the generated query"
                return False
            if len(self.variables) == len(self.__gold_query.variables):
                if self.uris == self.__gold_query.uris:
                    for permutation in self.__generate_all_permutations(self.gold_query):
                        if prepareQuery(permutation).algebra == prepareQuery(self.gold_query.sparql_query).algebra:
                            return True
                    return False
                else:
                    return False
            else:
                return False
        except:
            return False
    
    @property
    def confusion_matrix(self):
        return self._confusion_matrix

    @property
    def hallucinated_uris(self):
        return self._hallucinated_uris

    @hallucinated_uris.setter
    def hallucinated_uris(self, hallucinated_uris: set):
        self._hallucinated_uris = hallucinated_uris

    @confusion_matrix.setter
    def confusion_matrix(self, confusion_matrix: ConfusionMatrix):
        self._confusion_matrix = confusion_matrix

    def _has_triple_flip(self):
        
        try:
            all_permutations = self.__generate_all_permutations(self.__gold_query)
        except:
            self.logs = "<self._has_triple_flip> It was not possible to generate all permutations."
            return None

        for permutation in all_permutations:

            p_triples = SPARQLQuery(permutation).triples
            q_triples = self.__gold_query.triples

            try:
                if len(p_triples) != len(q_triples): return False
            except:
                return None

            new_query = permutation
            
            for p in range(len(p_triples)):
                flag = False
                for q in range(len(q_triples)):
                    p_subj, p_pred, p_obj = p_triples[p]
                    q_subj, q_pred, q_obj = q_triples[q]
                    if p_subj == q_obj and p_pred == q_pred and p_obj == q_subj:
                        new_query = SPARQLQuery(new_query)._flip_triple(p_triples[p])
                        flag = True
                if flag == True and (prepareQuery(new_query).algebra == prepareQuery(self.__gold_query.sparql_query).algebra):
                    return True
                                
        return False
    
    def __generate_all_permutations(self, q2):

        if not isinstance(q2, SPARQLQuery):
            raise ValueError("The input must be a SPARQLQuery object. Received: ", type(q2))

        all_permutations = []
        for variable_permutation in self.__generate_sparql_variable_permutations(q2):
            for triple_permutation in SPARQLQuery(variable_permutation).triple_permutations:
                all_permutations.append(self._standardize_query_to_parser(triple_permutation))
        
        return all_permutations
    
    def __generate_sparql_variable_permutations(self, q2):
    
        if not isinstance(q2, SPARQLQuery):
            raise ValueError("The input must be a SPARQLQuery object. Received: ", type(q2))
        
        if q2.has_query_parsing_error or self.has_query_parsing_error or self._sparql_query is None or q2.sparql_query is None:
            self.logs = "<__generate_sparql_variable_permutations> The queries have parsing errors. Therefore, it is not possible to generate variable permutations."
            return None

        v1 = list(self.variables)
        v2 = list(q2.variables)

        if len(v1) != len(v2):
            self.logs = "<__generate_sparql_variable_permutations> The number of variables in the queries are different. Therefore, it is not possible to generate variable permutations."
            return None
        
        all_permutations = list(permutations(v2))
        
        all_mappings = []
        for perm in all_permutations:
            mapping = [('?' + v1[i], '?' + perm[i]) for i in range(len(v1))]
            all_mappings.append(mapping)

        replaced_queries = []
        for mapping in all_mappings:
            mapping = {chave: valor for chave, valor in mapping}
            temps = {element: f"__temp_{i}__" for i, element in enumerate(mapping)}
        
            query_ = self._sparql_query

            for element, temp in temps.items():
                query_ = query_.replace(element, temp)
            
            for element, valor in mapping.items():
                query_ = query_.replace(temps[element], valor)

            replaced_queries.append(query_)

        return replaced_queries