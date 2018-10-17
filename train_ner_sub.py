from typing import List

import torch

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings, CharacterEmbeddings

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03).downsample(0.1)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings

embedding_types: List[TokenEmbeddings] = [
    CharLMEmbeddings('/cl/work/shusuke-t/flair_myLM/resources/taggers/language_model_sub/best-lm.pt'),
    CharLMEmbeddings('/cl/work/shusuke-t/flair_myLM/resources/taggers/language_model_sub_back/best-lm.pt'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# initialize trainer
from flair.trainers.sequence_tagger_trainer import SequenceTaggerTrainer

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

trainer.train('resources/taggers/ner_sub', learning_rate=0.1, mini_batch_size=32, max_epochs=150)
