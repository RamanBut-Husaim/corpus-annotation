from corpus_tokenizer import CorpusTokenizer
from corpus_statistics import CorpusStatistics
from corpus_statistics_provider import CorpusStatisticsProvider

def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return text

def display_corpus_statistics(corpus_statistics: CorpusStatistics):
    most_recurring_named_entity = corpus_statistics.most_recurring_named_entity

    print(f'Number of multi-word named entities: {corpus_statistics.multi_word_named_entities_count}')
    print(f"Number of lemmas 'devotchka': {corpus_statistics.devotchka_lemma_count}")
    print(f"Number of tokens with the stem 'milk': {corpus_statistics.milk_stem_count}")
    print(f'Most frequent entity type: {corpus_statistics.most_recurring_entity_type}')
    print(f"Most frequent named entity token: ('{most_recurring_named_entity[0]}', '{most_recurring_named_entity[1]}')")
    print(f'Most common non-English words: {corpus_statistics.most_common_non_english_words}')
    print('Correlation between NOUN and PROPN and named entities: {0:.2f}'.format(corpus_statistics.pearson_correlation))

if __name__ == '__main__':
    file_path = input()
    corpus = read_file(file_path)

    tokenizer = CorpusTokenizer()
    tokenizer.initialize()
    tokenization_result = tokenizer.tokenize(corpus)

    statistics_provider = CorpusStatisticsProvider()
    statistics_provider.initialize()
    corpus_statistics = statistics_provider.gather_statistics(tokenization_result)

    display_corpus_statistics(corpus_statistics)
