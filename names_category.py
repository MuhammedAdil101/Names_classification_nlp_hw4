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
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from io import open
import glob
import os

torch.manual_seed(1)

@DatasetReader.register('pos-tutorial')
class PosDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
    def text_to_instance(self, tokens: List[Token], tags: List[str] = None ) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}
        

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field
            
        

        return Instance(fields)
    
    
    def findFiles(self,path): 
        return glob.glob(path)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
            
        for filename in self.findFiles(file_path):
            
            with open(filename,encoding='utf-8') as f:
                for line in f:
                    word= line.strip().split('\n')
                    pairs=[word[0] + "###" + filename]
                    words, category = zip(*(pair.split("###") for pair in pairs))
                    yield self.text_to_instance([Token(word) for word in words], category)
    
                   
@Model.register('lstm-tagger')
class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))

        self.accuracy = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
    
if __name__ == "__main__":
    params = Params.from_file('./experiment.jsonnet')
    serialization_dir = tempfile.mkdtemp()
    model = train_model(params, serialization_dir)

    # Make predictions
    predictor = SentenceTaggerPredictor(model, dataset_reader=PosDatasetReader())
    tag_logits = predictor.predict("'Dovesky'")['tag_logits']
    print(tag_logits)
    tag_ids = np.argmax(tag_logits, axis=-1)
    print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])


    