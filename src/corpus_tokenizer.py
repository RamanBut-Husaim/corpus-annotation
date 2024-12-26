from corpus_tokenization_result import CorpusTokenizationResult

import re

import numpy as np
import pandas as pd
import spacy

class CorpusTokenizer:
    _nlp: spacy.Language = None
    _token_restricted_character_regex = None

    def __init__(self):
        self._token_restricted_character_regex = re.compile(r'[><_\\/*]')

    def initialize(self):
        self._nlp = spacy.load("en_core_web_sm")
        pass

    def tokenize(self, text: str) -> CorpusTokenizationResult:
        if self._nlp is None:
            raise AssertionError("nlp is not initialized")

        tokens = self._tokenize_text(text)

        return tokens

    def _tokenize_text(self, text: str) -> CorpusTokenizationResult:
        doc = self._nlp(text)

        matrix = np.array(['Token', 'Lemma', 'POS', 'Entity_type', 'IOB_tag'])

        for token in doc:
            if self._should_include_token(token):
                matrix = np.vstack(
                    [matrix, [token.text, token.lemma_, token.pos_, self._get_entity_type(token), token.ent_iob_]])

        tokens_df = self._to_pandas_dataframe(matrix)

        return CorpusTokenizationResult(tokens_df, doc.ents)

    def _should_include_token(self, token):
        match = self._token_restricted_character_regex.search(token.text)
        if match is not None:
            return False

        if self._is_empty_or_whitespace_or_separator(token.text):
            return False

        return True

    def _get_entity_type(self, token):
        if self._is_empty_or_whitespace_or_separator(token.ent_type_):
            return None

        return token.ent_type_

    @staticmethod
    def _is_empty_or_whitespace_or_separator(s):
        return s == "" or s.isspace()

    @staticmethod
    def _to_pandas_dataframe(matrix: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(matrix[1:], columns=matrix[0])
