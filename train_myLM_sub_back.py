from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
import torch.nn as nn
import torch


# are you training a forward or backward LM?
is_forward_lm = False 

# load the default character dictionary
dictionary: Dictionary = Dictionary.load('chars')

# get your corpus, process forward and at the character level

corpus = TextCorpus('./resources/tasks/wmt11_sub',
                     dictionary,
                     is_forward_lm,
                     character_level=True,
                     subword=True)

# instantiate your language model, set hidden size and number of layers
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=2048,
                               nlayers=1)

# train your language model
trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('resources/taggers/language_model_sub_back',
               sequence_length=250,
               mini_batch_size=100,
               max_epochs=100000)
