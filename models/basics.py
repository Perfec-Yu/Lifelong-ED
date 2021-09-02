import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
import math

class BilinearClassifier(nn.Module):
    def __init__(self,
                 in1_features,
                 in2_features,
                 hidden_features,
                 out_features,
                 dropout,
                 activation):
        super(BilinearClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(torch, activation)
        self.linear1 = nn.Linear(in1_features, hidden_features)
        self.linear2 = nn.Linear(in2_features, hidden_features)
        self.bilinear = FastBiliner(hidden_features, hidden_features, out_features)
    def forward(self, input1, input2):
        input1 = self.dropout(self.activation(self.linear1(input1)))
        input2 = self.dropout(self.activation(self.linear2(input2)))
        hidden = self.bilinear(input1, input2)
        hidden_dims = len(hidden.size())
        permutation = list(range(hidden_dims-3)) + [hidden_dims-2, hidden_dims-1, hidden_dims-3]
        output = hidden.permute(*permutation)
        return output



class FastBiliner(nn.Module):
    def __init__(self, in1_features, in2_features, out_features):
        super(FastBiliner, self).__init__()
        weight = torch.randn(out_features, in1_features, in2_features) * math.sqrt(2 / (in1_features + in2_features))
        bias = torch.ones(out_features) * math.sqrt(2 / (in1_features + in2_features))
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.out_features = out_features
        self.in1_features = in1_features
        self.in2_features = in2_features
    def forward(self, input1, input2):
        # B x n x d
        assert len(input1.size()) == len(input2.size())
        input_dims = len(input1.size())
        weight_size = [1] * (input_dims-2) + list(self.weight.size())
        bias_size = [1] * (input_dims-2) + [self.out_features] + [1, 1]
        weight = self.weight.view(*weight_size)
        bias = self.bias.view(*bias_size)
        input1 = input1.unsqueeze(-3)
        input2 = input2.unsqueeze(-3).transpose(-2, -1)
        outputs = bias + torch.matmul(input1,
                                     torch.matmul(self.weight.unsqueeze(0),
                                                  input2))
        return outputs

class Linears(nn.Module):
    """Multiple linear layers with Dropout."""
    def __init__(self,
                 dimensions,
                 activation='relu',
                 dropout_prob=0.0,
                 bias=True):
        super(Linears, self).__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i],
                                               dimensions[i + 1],
                                               bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        inputs = self.dropout(inputs)
        return inputs

class BERTEncoder(nn.Module):
    def __init__(self,
                 bert_model="bert-large-cased",
                 output_layers=-1,
                 dropout=0.0,
                 cache_dir=None):
        super(BERTEncoder, self).__init__()
        bert_config = BertConfig.from_pretrained(bert_model,
                                                 cache_dir=cache_dir,
                                                 output_hidden_states=True)
        self.bert = BertModel.from_pretrained(bert_model,
                                              cache_dir=cache_dir,
                                              config=bert_config)
        self.hidden_size = bert_config.hidden_size
        self.drop = nn.Dropout(dropout)
        if not isinstance(output_layers, list):
            self.output_layers = [output_layers]
        else:
            self.output_layers = list(set([int(t) for t in output_layers]))

    def forward(self, input_ids, attention_mask):
        _, _, hidden = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask)
        output = torch.cat([hidden[layer] for layer in self.output_layers], -1)
        return self.drop(output)

    def merge_pieces(self, sequence, token_spans):
        return torch.matmul(token_spans, sequence)




