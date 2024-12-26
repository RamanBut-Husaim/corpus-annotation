class CorpusStatistics:
    _multi_word_named_entities_count: int = 0
    _devotchka_lemma_count: int = 0
    _milk_stem_count: int = 0
    _most_recurring_entity_type: str = ''
    _most_recurring_named_entity: tuple[str, str] = None
    _most_common_non_english_words: dict[str, int] = None
    _pearson_correlation: float = 0

    def __init__(self, multi_word_named_entities_count: int, devotchka_lemma_count: int, milk_stem_count: int,
                 most_recurring_entity_type: str, most_recurring_named_entity: tuple[str, str],
                 most_common_non_english_words: dict[str, int], pearson_correlation: float):
        self._multi_word_named_entities_count = multi_word_named_entities_count
        self._devotchka_lemma_count = devotchka_lemma_count
        self._milk_stem_count = milk_stem_count
        self._most_recurring_entity_type = most_recurring_entity_type
        self._most_recurring_named_entity = most_recurring_named_entity
        self._most_common_non_english_words = most_common_non_english_words
        self._pearson_correlation = pearson_correlation

    @property
    def multi_word_named_entities_count(self) -> int:
        return self._multi_word_named_entities_count

    @property
    def devotchka_lemma_count(self) -> int:
        return self._devotchka_lemma_count

    @property
    def milk_stem_count(self) -> int:
        return self._milk_stem_count

    @property
    def most_recurring_entity_type(self) -> str:
        return self._most_recurring_entity_type

    @property
    def most_recurring_named_entity(self) -> tuple[str, str]:
        return self._most_recurring_named_entity

    @property
    def most_common_non_english_words(self) -> dict[str, int]:
        return self._most_common_non_english_words

    @property
    def pearson_correlation(self) -> float:
        return self._pearson_correlation