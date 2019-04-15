from __future__ import unicode_literals, print_function, division
from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy , Average
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder,PytorchSeq2VecWrapper
from torch.nn import LogSoftmax
from torch.nn.modules import NLLLoss
from io import open
import glob
import os
torch.manual_seed(1)


@DatasetReader.register('names-reader')
class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one word per line and its label 
    Doveyski , Russian.txt
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
    def text_to_instance(self, tokens: List[Token], label: str ) -> Instance:
        word_field = TextField(tokens, self.token_indexers)
        fields = {"word": word_field}
 
        if label:
        
            label_field = LabelField(label=label)
            fields["label"] = label_field
            
    
        return Instance(fields)
    
    
    def findFiles(self,path): 
        return glob.glob(path)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
            
        for filename in self.findFiles(file_path):
            
            with open(filename,encoding='utf-8') as f:
                for line in f:
                    word= line.strip().split('\n')
                    word=str(word[0])
                    yield self.text_to_instance([Token(ch) for ch in word], filename.split('\\')[1])    
                    
                    
@Model.register('names-classifier')
class WordClassifier(Model):
    def __init__(self,
                 char_embeddings: TextFieldEmbedder,
                 encoder:Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.char_embeddings = char_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = Average()
        self.m = LogSoftmax()#I used that in order to escape from loss to blow up
        self.loss = NLLLoss()#I used this from tutorial RNN classifier
    def forward(self,
                word: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(word)
        embeddings = self.char_embeddings(word)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if label is not None:
            output["loss"] = self.loss(self.m(tag_logits), label)
            pred = tag_logits.max(1)[1]#it is giving the maximum elements for each tensor inside pred and their indexes so we are taking indexes
            self.accuracy((torch.eq(pred,label).sum()).double()/len(label))
        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
    
    
if __name__ == "__main__":
    params = Params.from_file('./experiment.jsonnet')
    serialization_dir = tempfile.mkdtemp()
    model = train_model(params, serialization_dir)

    # Make predictions
    predictor = SentenceTaggerPredictor(model, dataset_reader=PosDatasetReader())
    tag_logits = predictor.predict("Doveyski")['tag_logits']
    print(tag_logits)
    tag_ids = np.argmax(tag_logits, axis=-1)
    print([model.vocab.get_token_from_index(tag_ids, 'labels')])

    shutil.rmtree(serialization_dir)

