import pandas as pd

class CorpusTokenizationResult:
    _tokens: pd.DataFrame = None
    _named_entities: tuple = None

    def __init__(self, tokens: pd.DataFrame, named_entities: tuple):
        self._tokens = tokens
        self._named_entities = named_entities

    @property
    def tokens(self) -> pd.DataFrame:
        return self._tokens

    @property
    def named_entities(self) -> tuple:
        return self._named_entities