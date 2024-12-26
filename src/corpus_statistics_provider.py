from corpus_statistics import CorpusStatistics
from corpus_tokenization_result import CorpusTokenizationResult

import nltk
import pandas as pd
from scipy.stats.stats import pearsonr

class CorpusStatisticsProvider:
    def initialize(self):
        nltk.download('words')

    def gather_statistics(self, corpus: CorpusTokenizationResult) -> CorpusStatistics:
        multi_word_entity_count = self._count_multi_word_named_entities(corpus.named_entities)
        devotcha_lemma_count = self._count_devotcha_lemma(corpus.tokens)
        milk_stem_count = self._count_milk_stem(corpus.tokens)
        most_recurring_entity_type = self._get_most_recurring_named_entity_type(corpus.tokens)
        most_recurring_named_entity = self._get_most_frequent_named_entity_token(corpus.tokens)
        most_common_non_english_words = self._get_most_common_non_english_words(corpus.tokens)
        pearson_correlation = self._calculate_pearson_correlation(corpus.tokens)

        statistics = CorpusStatistics(multi_word_entity_count, devotcha_lemma_count, milk_stem_count,
                                      most_recurring_entity_type, most_recurring_named_entity,
                                      most_common_non_english_words, pearson_correlation)

        return statistics

    @staticmethod
    def _count_multi_word_named_entities(named_entities: tuple) -> int:
        multi_word_entity_count = sum(1 for ent in named_entities if len(ent) > 1)

        return multi_word_entity_count

    @staticmethod
    def _count_devotcha_lemma(tokens: pd.DataFrame) -> int:
        devotcha_df = tokens[tokens.Lemma == 'devotchka']
        return devotcha_df.shape[0]

    @staticmethod
    def _count_milk_stem(tokens: pd.DataFrame) -> int:
        milk_df = tokens[tokens.Lemma == 'milk']
        return milk_df.shape[0]

    @staticmethod
    def _get_most_recurring_named_entity_type(tokens: pd.DataFrame) -> str:
        named_entity_type_stats = tokens.Entity_type.value_counts()
        return named_entity_type_stats.index[0]

    @staticmethod
    def _get_most_frequent_named_entity_token(tokens: pd.DataFrame) -> tuple[str, str]:
        named_entities_df = tokens[tokens.Entity_type.notna()]
        named_entity_token_stats = named_entities_df.Lemma.value_counts()
        most_frequent_named_entity_token_lemma = named_entity_token_stats.index[0]
        most_frequent_named_entity_token_entity_types = \
            named_entities_df.loc[named_entities_df.Lemma == most_frequent_named_entity_token_lemma, 'Entity_type']

        most_frequent_named_entity_token_most_frequent_entity_type = \
            most_frequent_named_entity_token_entity_types.value_counts().index[0]

        return most_frequent_named_entity_token_lemma, most_frequent_named_entity_token_most_frequent_entity_type

    @staticmethod
    def _get_most_common_non_english_words(tokens: pd.DataFrame) -> dict:
        words = nltk.corpus.words.words()
        suitable_pos_tags = ['ADJ', 'ADV', 'NOUN', 'VERB']

        suitable_tokens_df = tokens[
            (tokens.Lemma.str.len() > 4) & (tokens.POS.isin(suitable_pos_tags)) & (~(tokens.Lemma.isin(words)))]

        suitable_tokens_stats = suitable_tokens_df.Token.value_counts()
        stats_dict = suitable_tokens_stats.head(10).to_dict()

        return stats_dict

    @staticmethod
    def _calculate_pearson_correlation(tokens: pd.DataFrame) -> float:
        binary_pos = tokens.POS.apply(lambda x: 1 if x in ['NOUN', 'PROPN'] else 0)
        binary_entity_type = tokens.Entity_type.apply(lambda x: 1 if x is not None else 0)

        return pearsonr(binary_pos, binary_entity_type)[0]