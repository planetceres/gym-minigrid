import os

import gym
from gym import error, spaces, utils
import spacy
from spacy.language import Language
from spacy.vocab import Vocab

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
import itertools

DATA_DIR = '.'
GLOVE_DIR = os.path.join(DATA_DIR, 'glove')

# Load Stanford GloVe model in SpaCy
def load_glove():
    return Language().from_disk(GLOVE_DIR)

# Load SpaCy language model in SpaCy
def load_language_model(language_model):
    if language_model.startswith('glove'):
        model = load_glove()
    else:
        model = spacy.load(language_model)
    return model

# Get phrase embedding and  for a mission
def phrase_embedding(doc):
    if doc.has_vector:
        embed_phrase = doc.vector
        l2_norm = doc.vector_norm

    else:
        embed_phrase = None
        l2_norm = None

    return embed_phrase, l2_norm

# Get dependencies from mission text string
def dependency_parser(doc):
    heads = ''
    for token in doc:
        heads = heads + ' ' + token.head.text
    return heads

'''
If using gloVe vectors, dependency parser will not work properly

'''
class DependencyParser(gym.core.ObservationWrapper):
    def __init__(
                self,
                env,
                use_word_embeddings=False,
                use_phrase_embeddings=True,
                use_dependency_parser=True,
                language_model='en_core_web_md'):
        super().__init__(env)

        if language_model in ['glove50', 'en_core_web_sm', 'en_core_web_md']:
            self.language_model = language_model
        else:
            self.language_model = None

        if self.language_model:
            '''
            For possible language models see:
                -https://spacy.io/usage/models#available
                    - 'glove50'
                    - 'en_core_web_sm'
                    - 'en_core_web_md'
            '''
            self.nlp = load_language_model(language_model)


        # For phrase level embeddings
        if use_phrase_embeddings:
            self.use_phrase_embeddings = use_phrase_embeddings
            self.embed_phrase, self.embed_norm = phrase_embedding(self.nlp(self.env.mission))
            self.embed_dim = self.embed_phrase.shape
            self.norm_dim = 1

        # For phrase level embeddings of dependencies
        if use_dependency_parser:
            self.use_dependency_parser = use_dependency_parser
            deps = dependency_parser(self.nlp(self.env.mission))
            self.embed_dep, self.embed_dep_norm = phrase_embedding(self.nlp(deps))



    def observation(self, obs):
        return obs


class LanguageToken(gym.core.ObservationWrapper):
    def __init__(
                self,
                env,
                language_model='en_core_web_sm'):
        super().__init__(env)

        if language_model in ['glove50', 'en_core_web_sm', 'en_core_web_md']:
            self.language_model = language_model
        else:
            self.language_model = None

        if self.language_model:
            '''
            For possible language models see:
                -https://spacy.io/usage/models#available
                    - 'glove50'
                    - 'en_core_web_sm'
                    - 'en_core_web_md'
            '''
            self.nlp = load_language_model(language_model)


        # For phrase level embeddings
        if use_phrase_embeddings:
            self.use_phrase_embeddings = use_phrase_embeddings
            self.embed_phrase, self.embed_norm = phrase_embedding(self.nlp(self.env.mission))
            self.embed_dim = self.embed_phrase.shape
            self.norm_dim = 1

        # For phrase level embeddings of dependencies
        if use_dependency_parser:
            self.use_dependency_parser = use_dependency_parser
            deps = dependency_parser(self.nlp(self.env.mission))
            self.embed_dep, self.embed_dep_norm = phrase_embedding(self.nlp(deps))





    def observation(self, obs):
        return obs



class LanguageTokenizer:
    def __init__(
                self,
                text,
                maxStrLen=64,
                language_model='en_core_web_sm'):
        #super().__init__(env)

        self.input_text = text

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        if language_model in ['glove50', 'en_core_web_sm', 'en_core_web_md']:
            self.language_model = language_model
        else:
            self.language_model = None


        self.cachedStr = None
        self.cachedArray = None
        self.mission_tokenized = None


    def tokens(self):

        #image = obs['image']
        mission = self.input_text

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, "mission string too long"
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray


            self.mission_tokenized = self.cachedArray.flatten()
        print('mission tokenized:', self.mission_tokenized)

        return self.mission_tokenized
