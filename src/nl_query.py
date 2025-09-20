from enum import Enum
from transformers import BertTokenizer, BertModel
from persistence_utils import load_object
import torch
import logging
import pickle

class SupportedEmbedders(Enum):
    BERT = 'BERT'
    
class NLQuery:
    
    def __init__(self, nl_query: str):
        self._nl_query = nl_query
        self._embeddings = self.__embed()

    @property
    def nl_query(self):
        return self._nl_query
    
    @property
    def embeddings(self):
        return self._embeddings
    
    @property
    def get_bert_embedding(self):
        for embedding in self.embeddings:
            if embedding['embedder'] == SupportedEmbedders.BERT.value:
                return embedding['embedding']
        raise ValueError("BERT embedding not found")
    
    @nl_query.setter
    def nl_query(self, nl_query: str):
        self._nl_query = nl_query
        self._embeddings = self.__embed()

    def __embed(self):
        results = []
        for embedder in SupportedEmbedders:
            if embedder == SupportedEmbedders.BERT:
                bert_embeddings = self.BertUtils(self.nl_query).embedding
                results.append({'embedder': embedder.value, 'embedding': bert_embeddings})
        return results
    
    class BertUtils:

        def __init__(self, text: str, model_name: str = 'bert-base-uncased'):

            loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
            for logger in loggers:
                if "transformers" in logger.name.lower():
                    logger.setLevel(logging.ERROR)

            self._tokenizer = BertTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
            self._model = BertModel.from_pretrained(model_name)
            self.__encoding = self.tokenizer.batch_encode_plus([text], padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)
            self._attention_mask = self.encoding['attention_mask']
            self._token_ids = self.encoding['input_ids']
            self._tokens = self.tokenizer.convert_ids_to_tokens(self.token_ids[0])
            self._embedding = self.__get_embedding()

        @property
        def encoding(self):
            return self.__encoding

        @property
        def tokenizer(self):
            return self._tokenizer
        
        @property
        def model(self):
            return self._model
        
        @property
        def attention_mask(self):
            return self._attention_mask
        
        @property
        def token_ids(self):
            return self._token_ids
        
        @property
        def embedding(self):
            return self._embedding
        
        def __get_embedding(self):
            with torch.no_grad():
                outputs = self.model(input_ids=self.token_ids, attention_mask=self.attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                return cls_embedding