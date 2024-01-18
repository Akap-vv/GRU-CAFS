import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):

    def __init__(self, embedding,
                 hidden_size, num_layers=1,
                 bidirectional=True, LSTM=False):
        """Encoder part for encoding sequences.

            Parameters
            ----------
            embedding : nn.Embedding
                Embedding layer to use

            hidden_size : int
                Size of hidden dimension

            num_layers : int, default=1
                Number of recurrent layers to use

            bidirectional : boolean, default=False
                If True, use bidirectional recurrent layer

            LSTM : boolean, default=False
                If True, use LSTM instead of GRU
            """
        # Call super
        super(Encoder, self).__init__()
        # Set hidden size
        self.hidden_size = hidden_size
        # Set number of layers
        self.num_layers = num_layers
        # Set bidirectional
        self.bidirectional = bidirectional
        # Set LSTM
        self.LSTM = LSTM

        # Initialise layers
        self.embedding = embedding
        self.recurrent = (nn.LSTM if LSTM else nn.GRU)(
            input_size    = self.embedding.embedding_dim,
            # input_size    = 1 ,
            hidden_size   = self.hidden_size,
            num_layers    = self.num_layers,
            batch_first   = True,
            bidirectional = bidirectional
        )

        # Set embedding dimension
        # self.embedding_size = self.embedding.embedding_dim

    def forward(self, input, hidden=None):
        """Encode data

            Parameters
            ----------
            input : torch.Tensor
                Tensor to use as input

            hidden : torch.Tensor
                Tensor to use as hidden input (for storing sequences)

            Returns
            -------
            output : torch.Tensor
                Output tensor

            hidden : torch.Tensor
                Hidden state to supply to next input
            """
        # Initialise hidden if not given
        if hidden is None:
            hidden = self.initHidden(input)
            # import pdb;pdb.set_trace() #1*128*128 [0]
        # import pdb;pdb.set_trace() 
        # Get input as embedding
        # from pudb import set_trace;set_trace()
        embedded = self.embedding(input) #128*10*30  128*128*128
        # import pdb;pdb.set_trace() 
        # Pass through recurrent layer
        # input = torch.unsqueeze(input, dim=2).float()
        _, hidden = self.recurrent(embedded, hidden) #1*128*128
        # _, hidden = self.recurrent(input, hidden)
        # import pdb;pdb.set_trace()

        # if self.bidirectional:
        #     embedded = torch.cat((embedded, embedded), dim=2)
        # import pdb;pdb.set_trace() 
        # Return result
        return  embedded, hidden

    def initHidden(self, X):
        """Create initial hidden vector for data X

            Parameters
            ----------
            X : torch.Tensor,
                Tensor for which to create a hidden state

            Returns
            -------
            result : torch.Tensor
                Initial hidden tensor
            """
        if self.LSTM:
            return (torch.zeros(self.num_layers*(1 + int(self.bidirectional)),
                                X.shape[0], self.hidden_size, device=X.device),
                    torch.zeros(self.num_layers*(1 + int(self.bidirectional)),
                                X.shape[0], self.hidden_size, device=X.device))
        else:
            return  torch.zeros(self.num_layers*(1 + int(self.bidirectional)),
                                X.shape[0], self.hidden_size, device=X.device)
