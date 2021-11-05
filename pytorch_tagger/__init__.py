#!/usr/bin/python3

__author__ = "Paulius Danenas"
__maintainer__ = "Paulius Danenas"
__email__ = "danpaulius@gmail.com"

from .base import Base_BERT_CRF, Base_ELMO_CRF, BaseLstmCRFMixin, BaseAttentiveCRFMixin

class BERT_LSTM_CRF(Base_BERT_CRF, BaseLstmCRFMixin): pass
class BERT_Attentive_CRF(Base_BERT_CRF, BaseAttentiveCRFMixin): pass
class ELMO_LSTM_CRF(Base_ELMO_CRF, BaseLstmCRFMixin): pass
class ELMO_Attentive_CRF(Base_ELMO_CRF, BaseAttentiveCRFMixin): pass

