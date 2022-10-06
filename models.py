import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer



class Conceptual_Similarity(nn.Module):
    def __init__(self, bert_hidden_size=768, lstm_hidden_size=512, classifier_hidden_size=512, dropout=0.1, pooling="mean"):
        super(Conceptual_Similarity, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=bert_hidden_size, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_1 = nn.Linear(bert_hidden_size + 4*lstm_hidden_size, classifier_hidden_size)
        self.linear_2 = nn.Linear(classifier_hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.lstm_hidden_size = lstm_hidden_size
        self.pooling = pooling


    def split_bert_output(self, attn_mask, seg):
        # Find the indices x and y
        # X is the index of [SEP] in sentence 1 and y is the index of [SEP] in sentence 2
        x, y = 0, 0
        previous = -1
        for i in range(attn_mask.shape[0]): # tensor.shape[0] = seq_length
            if attn_mask[i].item() == 1 and seg[i].item() == 1 and attn_mask[previous].item() == 1 and seg[previous].item() == 0:
                x = previous
            if attn_mask[i].item() == 0 and seg[i].item() == 0 and attn_mask[previous].item() == 1 and seg[previous].item() == 1:
                y = previous
            previous += 1
        return x, y


    def pool_tensor(self, tensor):
        def sum_pooling(tensor):
            return torch.sum(tensor, dim=0)

        def mean_pooling(tensor):
            return sum_pooling(tensor) / tensor.shape[0]

        def max_pooling(tensor):
            return torch.max(tensor, dim=0).values

        if self.pooling == "sum":
            pooling_function = sum_pooling
        if self.pooling == "mean":
            pooling_function = mean_pooling
        if self.pooling == "max":
            pooling_function = max_pooling
        
        return pooling_function(tensor)

        


    def forward(self, input, attn_mask=None, seg=None, h_0=None, c_0=None):
        # Get BERT's last hidden states for both inputs (sentences)
        output = self.model(input, attention_mask=attn_mask, token_type_ids=seg)[0] # To only get the last hidden state (batch_size, seq_length, hidden_size)

        # Get the [CLS] vector
        cls_vector = output[:, 0, :] # Because If I don't clone them, they will change (batch_size, hidden_size)

        concatenation = torch.zeros(output.shape[0], output.shape[-1] + 4 * self.lstm_hidden_size) # (batch_size, bert_hidden_size + 2 * lstm_hidden_size)

        # Element-wise multiplication
        for batch_idx in range(output.shape[0]):
            # Get the index spans of s1 and s2
            x, y = self.split_bert_output(attn_mask[batch_idx], seg[batch_idx])
            # Split the output into s1 and s2
            s1 = output[batch_idx,1:x,:] # (s1_seq_length, hidden_size)
            s2 = output[batch_idx,x+1:y,:] # (s2_seq_length, hidden_size)
            # Get a vector representation of s1 and s2
            s1_pooled = self.pool_tensor(s1) # (hidden_size)
            s2_pooled = self.pool_tensor(s2) # (hidde,_size)
            # Do the element-wise multiplication
            s1 = output[batch_idx,1:x,:] * s2_pooled # (s1_seq_length, hidden_size)
            s2 = output[batch_idx,x+1:y,:] * s1_pooled # (s2_seq_length, hidden_size)

            s1 = self.lstm(s1.unsqueeze(0), (h_0[:,batch_idx,:].unsqueeze(1), c_0[:,batch_idx,:].unsqueeze(1)))[1][0]  # Of size (2, 1, lstm_hidden_size) because bidirectional
            s2 = self.lstm(s2.unsqueeze(0), (h_0[:,batch_idx,:].unsqueeze(1), c_0[:,batch_idx,:].unsqueeze(1)))[1][0]  # 1 for batch_size = 1

            # Concatenate cls_vector, s1 and s2
            concatenation[batch_idx] = torch.cat((cls_vector[batch_idx], s1[0,0,:], s1[1,0,:], s2[0,0,:], s2[1,0,:]), 0)


        # Run through the classification NN
        return self.sigmoid(self.linear_2(self.relu(self.dropout(self.linear_1(concatenation)))))










##########################################################################
##########################################################################
##########################################################################
################                                          ################
################        Models used for evaluation        ################
################                                          ################
##########################################################################
##########################################################################
##########################################################################





class BERT_classification(nn.Module):
    def __init__(self, bert_hidden_size=768, classifier_hidden_size=512, dropout=0.1):
        super(BERT_classification, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.linear_1 = nn.Linear(bert_hidden_size, classifier_hidden_size)
        self.linear_2 = nn.Linear(classifier_hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)



    def forward(self, input, attn_mask=None, seg=None):
        last_hidden_state = self.model(input, attention_mask=attn_mask, token_type_ids=seg)[0] # [batch_size, seq_length, hidden_size]
        classification_vector = last_hidden_state[:, 0, :] # [batch_size, hidden_size]
        return self.sigmoid(self.linear_2(self.relu(self.dropout(self.linear_1(classification_vector)))))






class Siamese(nn.Module):
    def __init__(self, embedding_size=300, gru_hidden_size=256, classifier_hidden_size=512, dropout=0.1):
        super(Siamese, self).__init__()
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=gru_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.linear_1 = nn.Linear(3 * gru_hidden_size, classifier_hidden_size)
        self.linear_2 = nn.Linear(classifier_hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, input_1, input_2, h_0=None):
        input_1 = self.gru(input_1, h_0)[1].squeeze(0)  # Of size (1, gru_hidden_size) because bidirectional
        input_2 = self.gru(input_2, h_0)[1].squeeze(0)

        diff = torch.abs(input_1 - input_2)
        concat = torch.cat((diff, input_1, input_2), -1)  # Of size (1, 3 * gru_hidden_size)

        return self.sigmoid(self.linear_2(self.relu(self.dropout(self.linear_1(concat))))).squeeze()







