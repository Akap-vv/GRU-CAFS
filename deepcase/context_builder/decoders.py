import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderAttention(nn.Module):

    def __init__(self, embedding, context_size, attention_size, num_layers=1,
                 dropout=0.1, bidirectional=True, LSTM=False):
        """Attention decoder for retrieving attention from context vector.

            Parameters
            ----------
            embedding : nn.Embedding
                Embedding layer to use.

            context_size : int
                Size of context to expect as input.

            attention_size : int
                Size of attention vector.

            num_layers : int, default=1
                Number of recurrent layers to use.

            dropout : float, default=0.1
                Default dropout rate to use.

            bidirectional : boolean, default=False
                If True, use bidirectional recurrent layer.

            LSTM : boolean, default=False
                If True, use LSTM instead of GRU.
            """
        # Call super
        super().__init__()

        ################################################################
        #                      Initialise layers                       #
        ################################################################
        # Embedding layer
        self.embedding = embedding #Embedding(30, 128)

        # Recurrency layer
        self.recurrent = (nn.LSTM if LSTM else nn.GRU)(
            input_size    = embedding.embedding_dim, #128
            hidden_size   = context_size, #128
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = bidirectional,
        )

        # Attention layer
        self.attn = nn.Linear(
            in_features  = context_size * num_layers * (1+bidirectional),
            out_features = attention_size
            )
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, context_vector, previous_input=None):
        """Compute attention based on input and hidden state.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, embedding_dim)
                Input from which to compute attention

            hidden : torch.Tensor of shape=(n_samples, hidden_size)
                Context vector from which to compute attention

            Returns
            -------
            attention : torch.Tensor of shape=(n_samples, context_size)
                Computed attention

            context_vector : torch.Tensor of shape=(n_samples, hidden_size)
                Updated context vector
            """
        # Get embedding from input
        # from pudb import set_trace; set_trace()
        embedded = self.embedding(previous_input)\
                   .view(-1, 1, self.embedding.embedding_dim) #128*1*128
        # import pdb;pdb.set_trace()
        # Apply dropout layer
        embedded = self.dropout(embedded)

        # Compute attention and pass through hidden to next state
        attention, context_vector = self.recurrent(embedded, context_vector)
        # attention 128*1*128 ;;; context_vector 1*128*128
        # Apply dropout layer
        # attention = self.dropout(attention) 128*1*128
        # Compute attention
        # import pdb;pdb.set_trace()
        attention = self.attn(attention.squeeze(1))  #128*10
        # import pdb;pdb.set_trace()
        # Normalise attention weights, i.e. sum to 1
        attention = F.softmax(attention, dim=1) #按行归一化
        # import pdb;pdb.set_trace()

        # Return result
        return attention, context_vector



class DecoderEvent(nn.Module):

    def __init__(self, input_size, output_size, dropout=0.1):
        """"""
        # Call super
        super().__init__()

        # Initialise layers
        self.hidden  = nn.Linear(input_size, input_size)
        self.out     = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, attention):
        """Decode X with given attention.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, context_size, hidden_size)
                Input samples on which to apply attention.

            attention : torch.Tensor of shape=(n_samples, context_size)
                Attention to use for decoding step

            Returns
            -------
            output : torch.Tensor of shape=(n_samples, output_size)
                Decoded output
            """
        # Apply attention (by computing batch matrix-matrix product)
        attn = attention.unsqueeze(1)
        # import pdb;pdb.set_trace()
        # from pudb import set_trace;set_trace()
        attn_applied = torch.bmm(attn, X).squeeze(1) #128*1*10 128*10*30 ==>128*30
        # import pdb;pdb.set_trace()
        # attn_applied = self.dropout(attn_applied)

        # Compute prediction based on latent dimension
        output = self.hidden(attn_applied).relu()
        # output = self.dropout(output)
        output = self.out(output)  #128*30
        # import pdb;pdb.set_trace()
        # Apply softmax for distribution
        output = F.log_softmax(output, dim=1)
        # import pdb;pdb.set_trace()

        # Return result
        return output,attn_applied

class DecoderClassifier(nn.Module):

    def __init__(self, input_size, class_size, dropout=0.1):
        """"""
        # Call super
        super().__init__()

        # Initialise layers
        self.hidden  = nn.Linear(input_size, input_size)
        self.out     = nn.Linear(input_size, class_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, attention):
        """Decode X with given attention.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, context_size, hidden_size)
                Input samples on which to apply attention.

            attention : torch.Tensor of shape=(n_samples, context_size)
                Attention to use for decoding step

            Returns
            -------
            output : torch.Tensor of shape=(n_samples, output_size)
                Decoded output
            """
        # Apply attention (by computing batch matrix-matrix product)
        attn = attention.unsqueeze(1)
        # import pdb;pdb.set_trace()
        attn_applied = torch.bmm(attn, X).squeeze(1) #128*1*10 128*10*30 ==>128*30
        # import pdb;pdb.set_trace()
        # attn_applied = self.dropout(attn_applied)

        # Compute prediction based on latent dimension
        output = self.hidden(attn_applied).relu()
        # output = self.dropout(output)
        output = self.out(output)  #128*30
        # import pdb;pdb.set_trace()
        # Apply softmax for distribution
        output = F.log_softmax(output, dim=1)
        # import pdb;pdb.set_trace()

        # Return result
        return  output