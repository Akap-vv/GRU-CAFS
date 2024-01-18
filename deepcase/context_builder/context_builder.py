# Imports
import logging
import math
import random
from tqdm import tqdm
import numpy as np
import copy
from collections import Counter
# Torch imports
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim
from torch.autograd   import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
# Custom package imports
# import sys
# import os
# # sys.path.append('.')
# print(sys.path)
# print(os.getcwd() )
# print (sys.argv[0])
from .decoders  import DecoderAttention, DecoderEvent, DecoderClassifier
from .embedding import EmbeddingOneHot
from .encoders  import Encoder
from .loss      import LabelSmoothing
from .utils     import unique_2d

# from interpreter.utils   import group_by
def group_by(X, key=lambda x: x, verbose=False):
    """Group items based on their key function and return their indices.

        Parameters
        ----------
        X : array-like of shape=(n_samples,)
            Array for which to group elements.

        key : func, default=lambda x: x.item()
            Function used to return as group.

        verbose : boolean, default=False
            If True, print progress.

        Returns
        -------
        result : list of (group, indices)
            Where:
             - group  : object
                Group computed based on key(x).
             - indices: np.array of shape=(n_group_items,)
                Inidices of items in X belonging to given group.
        """
    # Cast to numpy array
    X = np.asarray(X)

    # Initialise lookup table
    groups = dict()

    # Add progress bar if required
    if verbose: X = tqdm(X, desc="Lookup table")

    # Loop over items in table
    for index, label in enumerate(X):
        hashed = key(label)
        # Add label to lookup table if it does not exist
        if hashed not in groups:
            groups[hashed] = [key(label), list()]
        # Append item
        groups[hashed][1].append(index)

    # Return groups and indices
    return [(v1, np.asarray(v2)) for v1, v2 in groups.values()]
# Set logger
logger = logging.getLogger(__name__)

class ContextBuilder(nn.Module):

    def __init__(self, input_size,  output_size,class_size, hidden_size=128, num_layers=1,
                 max_length=10, bidirectional=True, LSTM=False):
        """ContextBuilder that learns to interpret context from security events.
            Based on an attention-based Encoder-Decoder architecture.

            Parameters
            ----------
            input_size : int
                Size of input vocabulary, i.e. possible distinct input items

            output_size : int
                Size of output vocabulary, i.e. possible distinct output items

            hidden_size : int, default=128
                Size of hidden layer in sequence to sequence prediction.
                This parameter determines the complexity of the model and its
                prediction power. However, high values will result in slower
                training and prediction times

            num_layers : int, default=1
                Number of recurrent layers to use

            max_length : int, default=10
                Maximum lenght of input sequence to expect

            bidirectional : boolean, default=False
                If True, use a bidirectional encoder and decoder

            LSTM : boolean, default=False
                If True, use an LSTM as a recurrent unit instead of GRU
            """
        logger.info("ContextBuilder.__init__")

        # Initialise super
        super().__init__()

        ################################################################
        #                      Initialise layers                       #
        ################################################################

        # Create embedding
        # self.embedding_encoder = nn.Embedding(1515, embedded_size)
        self.embedding         = nn.Embedding(input_size, hidden_size)
        self.embedding_one_hot = EmbeddingOneHot(input_size)

        # Create encoder
        self.encoder = Encoder(
            embedding     = self.embedding_one_hot,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            bidirectional = bidirectional,
            LSTM          = LSTM
        )

        # Create attention decoder
        self.decoder_attention = DecoderAttention(
            embedding      = self.embedding,
            context_size   = hidden_size,
            attention_size = max_length,
            num_layers     = num_layers,
            dropout        = 0.1,
            bidirectional  = bidirectional,
            LSTM           = LSTM,
        )

        # Create event decoder
        self.decoder_event = DecoderEvent(
            input_size  = input_size,
            output_size = output_size,
            dropout     = 0.1,
        )

        self.decoder_class = DecoderClassifier(
            input_size  = input_size,
            class_size = class_size,
            dropout     = 0.1,
        )



    ########################################################################
    #                        ContextBuilder Forward                        #
    ########################################################################

    def forward(self, X, y=None, label=None, steps=1, teach_ratio=0.5):
        """Forwards data through ContextBuilder.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_len)
                Tensor of input events to forward.

            y : torch.Tensor of shape=(n_samples, steps), optional
                If given, use value of y as next input with probability
                teach_ratio.

            steps : int, default=1
                Number of steps to predict in the future.

            teach_ratio : float, default=0.5
                Ratio of sequences to train that use given labels Y.
                The remaining part will be trained using the predicted values.

            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples, steps, output_size)
                The confidence level of each output event.

            attention : torch.Tensor of shape=(n_samples, steps, seq_len)
                Attention corrsponding to X given as (batch, out_seq, in_seq).
            """
        # logger.info("forward {} samples".format(X.shape[0]))

        ####################################################################
        #                   Perform check on events in X                   #
        ####################################################################

        # if X.max() >= self.embedding_one_hot.input_size:
        #     raise ValueError(
        #         "Expected {} different input events, but received input event "
        #         "'{}' not in expected range 0-{}. Please ensure that the "
        #         "ContextBuilder is configured with the correct input_size and "
        #         "output_size".format(
        #         self.embedding_one_hot.input_size,
        #         X.max(), #10
        #         self.embedding_one_hot.input_size-1,
        #     ))

        ####################################################################
        #                           Forward data                           #
        ####################################################################


        # Initialise results
        confidence = list()
        attention  = list()
        class_out = list()
        attn_X = list()

        # Get initial inputs of decoder
        decoder_input  = torch.zeros(
            size       = (X.shape[0], 1), #128*1
            dtype      = torch.long,
            device     = X.device,
        )

        # Encode input
        X_input,context_vector = self.encoder(X)
        # import pdb;pdb.set_trace()
        # Loop over all targets
        for step in range(steps):
            # Compute attention
            attention_, context_vector = self.decoder_attention(
                context_vector = context_vector,
                previous_input = decoder_input,
            )

            # import pdb;pdb.set_trace()

            # Compute event probability distribution
            confidence_,attn_X_ = self.decoder_event(
                X         = X_input, #128*10*30
                attention = attention_, #128*10
            )
            # from pudb import set_trace;set_trace()
            class_ = self.decoder_class(
                X         = X_input, #128*10*30
                attention = attention_, #128*10
            )

            # import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace()

            # Store confidence
            confidence.append(confidence_)#128*30
            # Store attention
            attention .append(attention_)#128*10
            class_out .append(class_)
            attn_X .append(attn_X_)

            # Detatch from history
            if y is not None and random.random() <= teach_ratio:
                decoder_input = y[:, step]
            else:
                decoder_input = confidence_.argmax(dim=1).detach().unsqueeze(1)
            # import pdb;pdb.set_trace()

        # Return result
        # import pdb;pdb.set_trace()
        return torch.stack(confidence, dim=1), torch.stack(attention, dim=1),torch.stack(class_out, dim=1),torch.stack(attn_X, dim=1)


    ########################################################################
    #                         Fit/predict methods                          #
    ########################################################################

    def fit(self, X_train,X_val, y_train,y_val, label_train,label_val, model_path, epochs=10,  batch_size=128, learning_rate=0.01,
            optimizer=optim.SGD, teach_ratio=0.5, verbose=True):
        """Fit the sequence predictor with labelled data

            Parameters
            ----------
            X : array-like of type=int and shape=(n_samples, context_size)
                Input context to train with.

            y : array-like of type=int and shape=(n_samples, n_future_events)
                Sequences of target events.

            epochs : int, default=10
                Number of epochs to train with.

            batch_size : int, default=128
                Batch size to use for training.

            learning_rate : float, default=0.01
                Learning rate to use for training.

            optimizer : optim.Optimizer, default=torch.optim.SGD
                Optimizer to use for training.

            teach_ratio : float, default=0.5
                Ratio of sequences to train including labels.

            verbose : boolean, default=True
                If True, prints progress.

            Returns
            -------
            self : self
                Returns self
            """
        # logger.info("fit {} samples".format(X1.shape[0]))

        # Get current mode
        mode = self.training
        # Get input as torch tensors
        device = next(self.parameters()).device
        X_train = torch.as_tensor(X_train, dtype=torch.int64, device=device) #391*10
        y_train = torch.as_tensor(y_train, dtype=torch.int64, device=device)  #391*1
        X_val = torch.as_tensor(X_val, dtype=torch.int64, device=device)  # 391*10
        y_val = torch.as_tensor(y_val, dtype=torch.int64, device=device)
        label_train = torch.as_tensor(label_train, dtype=torch.int64, device=device)
        label_val = torch.as_tensor(label_val, dtype=torch.int64, device=device)
        # import pdb; pdb.set_trace()
        # X, y, inverse = unique_2d(X, y)
        # same_id = group_by(inverse.cpu().numpy())
        # for inv_id, indices in same_id:
        #     label_id = label[indices]
        #     ind22,times22 = np.unique(label_id,return_counts=True)
        # import pdb; pdb.set_trace()

        # np.random.seed(2023)
        # rand = np.arange(0, X.shape[0], 1)
        # np.random.shuffle(rand)
        # X_train = X[rand[:int(X.shape[0]*0.8)]]
        # X_val = X[rand[int(X.shape[0]*0.8):]]
        # y_train = y[rand[:int(X.shape[0]*0.8)]]
        # y_val = y[rand[int(X.shape[0]*0.8):]]
        # label_train = label[rand[:int(X.shape[0]*0.8)]]
        # label_val = label[rand[int(X.shape[0]*0.8):]]
        # ind,times=np.unique(label_val.cpu().numpy(),return_counts=True)
        # import pdb;pdb.set_trace()
        # Set to training mode
        self.train()
        # import pdb;pdb.set_trace()
        # X_train, y_train, inverse = unique_2d(X_train, y_train)
        # import pdb;pdb.set_trace()
        # X_val, y_val, inverse = unique_2d(X_val, y_val)
        # import pdb;pdb.set_trace()
        # Set criterion and optimiser
        criterion1 = LabelSmoothing(self.decoder_event.out.out_features, 0.1)
        # criterion1 = nn.CrossEntropyLoss()
        criterion2 = LabelSmoothing(self.decoder_class.out.out_features, 0.1)
        # criterion = nn.MSELoss()
        
        best_accuracy=0
        # epoch_plen_loss = 1000
        # min_epo=0
        # single_loss = False
        # LEARN_RATE = [0.01,0.001]
        # BATCH_SIZE = [128,256]
        # Loop over each epoch for packet length prediction
        # for learning_rate in LEARN_RATE:
        #     for batch_size in BATCH_SIZE:
        #         print('lr = {}, Batch_size= {}'.format(learning_rate,batch_size))
        optimizer = optim.SGD(
            params = self.parameters(),
            lr     = learning_rate,
        )
        # from pudb import set_trace;set_trace()
        # import pdb;pdb.set_trace()
        data = DataLoader(TensorDataset(X_train, y_train,label_train),
            batch_size = batch_size,
            shuffle    = True,
        )
        for epoch in range(1, epochs+1):
            try:
                # Set progress bar if necessary
                if verbose:
                    data = tqdm(data,
                        desc="[Epoch {:{width}}/{:{width}} loss={:.4f}]"
                        .format(epoch, epochs, 0, width=len(str(epochs)))
                    )

                # Set average loss
                total_loss  = 0
                total_items = 0

                # Loop over entire dataset
                for X_, y_ ,label_ in data:
                    # Clear gradients
                    optimizer.zero_grad()
                    # import pdb;pdb.set_trace()
                    # Get prediction
                    confidence, _, class_out,_ = self.forward(X_, y_,label_,
                        steps       = y_.shape[1],
                        teach_ratio = teach_ratio
                    )
                    # import pdb;pdb.set_trace()
                    # Compute loss
                    loss = 0
                    for step in range(confidence.shape[1]):
                        # import pdb;pdb.set_trace()
                        loss1 = criterion1(confidence[:, step], y_[:, step].long())
                        # loss1 = nn.CrossEntropyLoss(confidence[:, step].squeeze(1), y_[:, step])
                        loss2 = criterion2(class_out[:, step], label_[:, 0])
                        # import pdb;pdb.set_trace()
                        loss = loss1*0+loss2
                    loss.backward()
                    optimizer.step()

                    # Update description
                    total_loss  += loss.item() / X_.shape[1]
                    total_items += X_.shape[0]
                    # total_loss  += loss.item() / X_.shape[1]
                    # total_items_plen += X_.shape[0]

                    if verbose:
                        data.set_description(
                            "[Epoch {:{width}}/{:{width}} loss={:.4f}]"
                            .format(epoch, epochs, total_loss/total_items,
                            width=len(str(epochs))))
            except KeyboardInterrupt as e:
                print("\nTraining interrupted, performing clean stop")
                break       
        
            with torch.no_grad():
                # import pdb;pdb.set_trace()
                confidence_val, _, class_out_val,_ = self.forward(X_val, steps=1)
                # import pdb;pdb.set_trace()
                class_out_val = class_out_val[:, 0].exp()
                class_out_val = class_out_val.argmax(dim=1)
                # confidence_val = confidence_val[:, 0].exp()
                # confidence_val = confidence_val.argmax(dim=1)
                accuracy = accuracy_score(label_val.cpu(), class_out_val.cpu())*100
                # accuracy2 = accuracy_score(y_val.cpu(), confidence_val.cpu()) * 100
                # accuracy = (accuracy2 + accuracy1)/2
                print('val_accuracy: {}'.format(accuracy))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    # self.save('../model/2012filter_lr{}_bs{}_acc{}.pth'.format(learning_rate,batch_size,int(best_accuracy*100)))
                    self.save('../model/'+model_path)
                    print('save-------   '+ str(accuracy.item()) )

        self.train(mode)

        return self

    # def fit_kv(self, X, y):
    #     from sklearn.model_selection import KFold


    #     kfold = KFold(n_splits=5, shuffle=True)


    #     for train_idxs, val_idxs in kfold.split(X):

    #         X_train, y_train = X[train_idxs], y[train_idxs]
    #         X_val, y_val = X[val_idxs], y[val_idxs]

    #         self.fit(X_train, y_train, epochs=10, batch_size=128, learning_rate=0.01,
    #                 optimizer=optim.SGD, teach_ratio=0.5, verbose=True)

    #         loss, accuracy = model.evaluate(X_val, y_val)
    #         print("Validation loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

    def predict(self, X, y=None,steps=1):
        """Predict the next elements in sequence.

            Parameters
            ----------
            X : torch.Tensor
                Tensor of input sequences

            y : ignored

            steps : int, default=1
                Number of steps to predict into the future

            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples, seq_len, output_size)
                The confidence level of each output

            attention : torch.Tensor of shape=(n_samples, input_length)
                Attention corrsponding to X given as (batch, out_seq, seq_len)
            """
        # logger.info("predict {} samples".format(X.shape[0]))
        
        # Get current mode
        mode = self.training
        # Set to prediction mode
        self.eval()

        # Memory optimization, only use unique values
        device = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.int64, device=device)
        X, inverse = torch.unique(X, dim=0, return_inverse=True)
        # # import pdb;pdb.set_trace()

        # logger.info("predict {}/{} unique samples".format(X.shape[0], inverse.shape[0]))
        

        # Do not perform gradient descent
        with torch.no_grad():
            # Perform all in single batch
            confidence, attention, class_out,attn_X = self.forward(X, steps=steps)
        attn_X = attn_X.squeeze(1)
        # import pdb;pdb.set_trace()
        # from pudb import set_trace;set_trace()
        # Reset to original mode
        self.train(mode)
        if steps==1:
            class_out = class_out[:, 0]
            class_out = class_out.exp()
            class_out = class_out.argmax(dim=1)
            confidence = confidence[:, 0]
            confidence = confidence.exp()
            confidence = confidence.argmax(dim=1)
            return confidence[inverse].cpu().numpy(), attention[inverse], class_out[inverse].cpu().numpy(),attn_X[inverse]
        else:
            class_out = class_out.exp()
            class_out = class_out.argmax(dim=2)
            class_eq = list()
            for i in range(class_out.shape[0]):
                class_eq.append(Counter(class_out[i]).most_common(1)[0][0])
            class_eq = np.array(class_eq)
            # from pudb import set_trace;set_trace()
            confidence = confidence.exp()
            confidence = confidence.argmax(dim=2)
        # Return result
            return confidence[inverse].cpu().numpy(), attention[inverse], class_eq[inverse]
        # return confidence, attention, class_out

    def predict1(self, X, y=None, steps=1):
        """Predict the next elements in sequence.

            Parameters
            ----------
            X : torch.Tensor
                Tensor of input sequences

            y : ignored

            steps : int, default=1
                Number of steps to predict into the future

            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples, seq_len, output_size)
                The confidence level of each output

            attention : torch.Tensor of shape=(n_samples, input_length)
                Attention corrsponding to X given as (batch, out_seq, seq_len)
            """
        # logger.info("predict {} samples".format(X.shape[0]))

        mode = self.training
        # Set to prediction mode
        self.eval()

        # Memory optimization, only use unique values
        device = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.int64, device=device)
        X, inverse = torch.unique(X, dim=0, return_inverse=True)
        # X2 = torch.as_tensor(X2, dtype=torch.int64, device=device)
        # X1, inverse = torch.unique(X1, dim=0, return_inverse=True)
        # # import pdb;pdb.set_trace()

        # logger.info("predict {}/{} unique samples".format(X.shape[0], inverse.shape[0]))

        # Do not perform gradient descent
        with torch.no_grad():
            # Perform all in single batch
            confidence, attention, class_out,_ = self.forward(X, steps=steps)

        # Reset to original mode
        self.train(mode)

        # Return result
        # return confidence[inverse], attention[inverse]
        return confidence[inverse], attention[inverse], class_out[inverse]

    def pre_train(self, X, input_size,batch_size=128, y=None, steps=1):
        """Predict the next elements in sequence.

            Parameters
            ----------
            X : torch.Tensor
                Tensor of input sequences

            y : ignored

            steps : int, default=1
                Number of steps to predict into the future

            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples, seq_len, output_size)
                The confidence level of each output

            attention : torch.Tensor of shape=(n_samples, input_length)
                Attention corrsponding to X given as (batch, out_seq, seq_len)
            """
        # logger.info("predict {} samples".format(X.shape[0]))
        
        # Get current mode
        mode = self.training
        # Set to prediction mode
        self.eval()
        device = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.int64, device=device) #391*10
        # import pdb;pdb.set_trace()
        data = DataLoader(TensorDataset(X),
            batch_size = batch_size,
            shuffle    = False,
        )
        attention = torch.empty((0, input_size),
                                device = device,
                                )

        # X1, inverse = torch.unique(X1, dim=0, return_inverse=True)
        
        # logger.info("predict {}/{} unique samples".format(X.shape[0], inverse.shape[0]))
        

        # Do not perform gradient descent
        with torch.no_grad():
            for X_ in data:
                # Perform all in single batch
                
                _, attention_ = self.forward(X_[0], steps=steps)
                # import pdb;pdb.set_trace()
                attention = torch.cat([attention, attention_.squeeze(1)], dim=0)
                # import pdb;pdb.set_trace()

        # Reset to original mode
        self.train(mode)

        # Return result
        # return confidence[inverse], attention[inverse]
        return  attention

    def fit_predict(self, X, y, epochs=10, batch_size=128, learning_rate=0.01,
                    optimizer=optim.SGD, teach_ratio=0.5, verbose=True):
        """Fit the sequence predictor with labelled data

            Parameters
            ----------
            X : torch.Tensor
                Tensor of input sequences

            y : torch.Tensor
                Tensor of output sequences

            epochs : int, default=10
                Number of epochs to train with

            batch_size : int, default=128
                Batch size to use for training

            learning_rate : float, default=0.01
                Learning rate to use for training

            optimizer : optim.Optimizer, default=torch.optim.SGD
                Optimizer to use for training

            teach_ratio : float, default=0.5
                Ratio of sequences to train including labels

            verbose : boolean, default=True
                If True, prints progress

            Returns
            -------
            result : torch.Tensor
                Predictions corresponding to X
            """
        logger.info("fit_predict {} samples".format(X.shape[0]))

        # Apply fit and predict in sequence
        return self.fit(
            X             = X,
            y             = y,
            epochs        = epochs,
            batch_size    = batch_size,
            learning_rate = learning_rate,
            optimizer     = optimizer,
            teach_ratio   = teach_ratio,
            verbose       = verbose,
        ).predict(X)

    ########################################################################
    #                         ContextBuilder Query                         #
    ########################################################################

    def query(self, X, y, label,model_path,iterations=0, batch_size=1024, ignore=None,
              return_optimization=0.2, verbose=True):
        """Query the network to get optimal attention vector.

            Parameters
            ----------
            X : array-like of type=int and shape=(n_samples, context_size)
                Input context of events, same as input to fit and predict

            y : array-like of type=int and shape=(n_samples,)
                Observed event

            iterations : int, default=0
                Number of iterations to perform for optimization of actual event

            batch_size : int, default=1024
                Batch size of items to optimize

            ignore : int, optional
                If given ignore this index as attention

            return_optimization : float, optional
                If given, returns number of items with confidence level larger
                than given parameter. E.g. return_optimization=0.2 will also
                return two boolean tensors for elements with a confidence >= 0.2
                before optimization and after optimization.

            verbose : boolean, default=True
                If True, print progress

            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples, output_size)
                Confidence of each prediction given new attention

            attention : torch.Tensor of shape=(n_samples, context_size)
                Importance of each input with respect to output

            inverse : torch.Tensor of shape=(n_samples,)
                Inverse is returned to reconstruct the original array

            confidence_orig : torch.Tensor of shape=(n_samples,)
                Only returned if return_optimization != None
                Boolean array of items >= threshold before optimization

            confidence_optim : torch.Tensor of shape=(n_samples,)
                Only returned if return_optimization != None
                Boolean array of items >= threshold after optimization
            """
        # Get device
        device = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.int64, device=device)  # 391*10
        y = torch.as_tensor(y, dtype=torch.int64, device=device)
        label = torch.as_tensor(label, dtype=torch.int64, device=device)
        original_device = X.device
        # from pudb import set_trace;set_trace()
        # Initialise result
        result_confidence = list()
        result_attention  = list()
        # import pdb;pdb.set_trace()
        # Memory optimization, only use unique values
        X, y, label, inverse = unique_2d(X, y,label)
        # import pdb;pdb.set_trace()
        # str1 = copy.deepcopy(self)
        # Ignore given datapoints
        if ignore is not None:
            raise NotImplementedError("Ignore is not properly implemented yet.")
            attention[X == ignore] = 0

        # Squeeze variables
        y = y.squeeze(1)
        label = label.squeeze(1)
        # import pdb;pdb.set_trace()

        # for name, p in self.named_parameters():
        #     if "attention" in name:
        #         print("update only", name)
        #         p.requires_grad = True
        #
        # params = filter(lambda p: p.requires_grad, self.parameters())
        # Initialise progress if necessary
        if verbose:
            progress = tqdm(None,
                total = int(iterations)*int(math.ceil(X.shape[0]/batch_size)),
                desc  = "Optimizing query",
            )
        # from pudb import set_trace;set_trace()
        # Batch data
        batches = DataLoader(
            TensorDataset(X, y,label),
            batch_size = batch_size,
            shuffle    = False,
        )

        # Count datapoints with confidence >= 0.2
        if return_optimization is not None:
            confidence_orig  = list()
            confidence_optim = list()
            classify_orig = list()
            classify_optim = list()

        ################################################################
        #                    Attention optimisation                    #
        ################################################################

        # Loop over batches
        for batch, (X_, y_,label_) in enumerate(batches):
            # Compute initial attention and confidence
            # import pdb;pdb.set_trace()
            confidence, attention, class_out = self.predict1(X_, y_,)

            confidence = confidence.squeeze(1) #128*1*100-->128*100
            attention  = attention .squeeze(1) #128*1*10-->128*10
            classify = class_out.squeeze(1)
            # import pdb;pdb.set_trace()

            # Count confidence >= 0.2 of non-optimized datapoints
            if return_optimization is not None:

                confidence_orig.append((
                ##将 log_softmax() 函数输出的概率转换为实际的概率值，应用指数函数 exp()
                    confidence[torch.arange(y_.shape[0]), y_].exp() >= return_optimization
                ).detach().clone())
                classify_orig.append(classify.detach().clone())
            # import pdb;pdb.set_trace()

            # Make attention variable
            attn = attention
            # import pdb;pdb.set_trace()
            # Set optimizer

            for name, p in self.named_parameters():
                if "attention" not in name:
                    # print("update only", name)
                    p.requires_grad = False

            params = filter(lambda p: p.requires_grad, self.parameters())
            # for name, value in self.named_parameters():
            #     print(name, value.requires_grad)  # 打印所有参数requires_grad属性，True或False
            optimizer = optim.Adam(
                params=params,
                lr=0.001,
            )
            # import pdb;pdb.set_trace()
            # optimizer = optim.Adam([attn], lr=0.1)
            criterion = nn.NLLLoss()
            decoder_input = torch.zeros(
                size=(X_.shape[0], 1),  # 128*1
                dtype=torch.long,
                device=X_.device,
            )
            # Encode values of X
            with torch.no_grad():
                X_, context_vector = self.encoder(X_)
            loss_best=1
            # Perform iterations
            for iteration in range(int(iterations)):
                # Clear optimizer
                optimizer.zero_grad()

                # Add decoding function
                def decode(input, attn, softmax=False):
                    if softmax: attn,_ = self.decoder_attention(
                        context_vector=context_vector,
                        previous_input=decoder_input,
                    )
                    confi,_ = self.decoder_event(input, attn)
                        # attn = F.softmax(attn, dim=1)
                    return confi,self.decoder_class(input, attn)

                # from pudb import set_trace;set_trace()
                # Perform prediction
                attn1 = attn
                pred_confi, pred_label= decode(X_, attn, softmax=iteration > 0)
                # import pdb;pdb.set_trace()

                # Compute loss
                loss1 = criterion(pred_confi, y_)
                loss2 = criterion(pred_label, label_)
                loss = loss1*0.25+loss2
                # if loss < loss_best:
                #     loss_best = loss
                #     self.save('../model/' + model_path)
                print(loss1,loss2)

                # Perform backpropagation
                loss.requires_grad_(True)
                loss.backward()
                # import pdb; pdb.set_trace()
                optimizer.step()



                # Update progress if necessary
                if verbose: progress.update()

            # Perform final softmax
            if iterations > 0: attn,_ = self.decoder_attention(
                        context_vector=context_vector,
                        previous_input=decoder_input,
                    )

            # Detach attention - memory optimization
            attn = attn.detach()

            # Get confidence levels
            confidence_,_ = self.decoder_event(X_, attn)
            # str2 = copy.deepcopy(self)
            # import pdb;pdb.set_trace()
            confidence_ = confidence_[torch.arange(y_.shape[0]), y_].exp().detach()
            # import pdb;pdb.set_trace()
            confidence  = confidence [torch.arange(y_.shape[0]), y_].exp().detach()
            # import pdb;pdb.set_trace()

            # Check where confidence improved
            mask = confidence_ > confidence

            # Store attention if we improved
            attention[mask] = attn[mask]
            # import pdb;pdb.set_trace()

            # Recompute confidence
            with torch.no_grad():
                confidence,_ = self.decoder_event(
                    X         = X_,
                    attention = attention,
                )
                confidence = confidence.exp()
                classify = self.decoder_class(
                    X=X_,
                    attention=attention,
                )
                # str3 = str(self.state_dict())
                # import pdb; pdb.set_trace()
                # Count confidence >= 0.2 of optimized datapoints
                if return_optimization is not None:
                    confidence_optim.append((
                        confidence[torch.arange(y_.shape[0]), y_] >= return_optimization
                    ).detach().clone())
                    classify_optim.append(classify.detach().clone())


            # Add confidence and attention to result
            result_confidence.append(confidence.cpu())
            result_attention .append(attention .cpu())

        # Combine confidence and attention into tensor
        # and cast to original device
        confidence = torch.cat(result_confidence).to(original_device)
        attention  = torch.cat(result_attention) .to(original_device)
        classify_orig = torch.cat(classify_orig)
        classify_optim = torch.cat(classify_optim)
        loss_orig = criterion(classify_orig, label)
        loss_optim = criterion(classify_optim, label)
        # Close progress if necessary
        if verbose: progress.close()
        # str2 = str(copy.deepcopy(self.state_dict()))
        # import pdb;pdb.set_trace()
        self.save('../model/'+model_path)
        # Return confidence optimization if necessary
        if return_optimization is not None:
            confidence_orig  = torch.cat(confidence_orig ).numpy()
            confidence_optim = torch.cat(confidence_optim).numpy()
            # Return result
            return confidence[inverse], attention[inverse],confidence_orig[inverse], confidence_optim[inverse],loss_orig,loss_optim

        # Return result
        return confidence, attention, inverse

    def query_seq(self, X, y, label,model_path, iterations=0, batch_size=1024, ignore=None,
              return_optimization=0.2, verbose=True):
        """Query the network to get optimal attention vector.

            Parameters
            ----------
            X : array-like of type=int and shape=(n_samples, context_size)
                Input context of events, same as input to fit and predict

            y : array-like of type=int and shape=(n_samples,)
                Observed event

            iterations : int, default=0
                Number of iterations to perform for optimization of actual event

            batch_size : int, default=1024
                Batch size of items to optimize

            ignore : int, optional
                If given ignore this index as attention

            return_optimization : float, optional
                If given, returns number of items with confidence level larger
                than given parameter. E.g. return_optimization=0.2 will also
                return two boolean tensors for elements with a confidence >= 0.2
                before optimization and after optimization.

            verbose : boolean, default=True
                If True, print progress

            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples, output_size)
                Confidence of each prediction given new attention

            attention : torch.Tensor of shape=(n_samples, context_size)
                Importance of each input with respect to output

            inverse : torch.Tensor of shape=(n_samples,)
                Inverse is returned to reconstruct the original array

            confidence_orig : torch.Tensor of shape=(n_samples,)
                Only returned if return_optimization != None
                Boolean array of items >= threshold before optimization

            confidence_optim : torch.Tensor of shape=(n_samples,)
                Only returned if return_optimization != None
                Boolean array of items >= threshold after optimization
            """
        # Get device
        device = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.int64, device=device)  # 391*10
        y = torch.as_tensor(y, dtype=torch.int64, device=device)
        label = torch.as_tensor(label, dtype=torch.int64, device=device)
        original_device = X.device
        # from pudb import set_trace;set_trace()
        # Initialise result
        result_confidence = list()
        result_attention = list()
        # import pdb;pdb.set_trace()
        # Memory optimization, only use unique values
        # X, y, inverse = unique_2d(X, y)
        # import pdb;pdb.set_trace()

        # Ignore given datapoints
        if ignore is not None:
            raise NotImplementedError("Ignore is not properly implemented yet.")
            attention[X == ignore] = 0

        # Squeeze variables
        y = y.squeeze(1)
        label = label.squeeze(1)
        # import pdb;pdb.set_trace()

        # Initialise progress if necessary
        if verbose:
            progress = tqdm(None,
                            total=int(iterations) * int(math.ceil(X.shape[0] / batch_size)),
                            desc="Optimizing query",
                            )
        # from pudb import set_trace;set_trace()
        # Batch data
        batches = DataLoader(
            TensorDataset(X, y, label),
            batch_size=batch_size,
            shuffle=False,
        )

        # Count datapoints with confidence >= 0.2
        if return_optimization is not None:
            confidence_orig = torch.empty((y.shape[0],y.shape[1]),
                                dtype=torch.int64,
                                device = device,
                                )
            confidence_optim = torch.empty((y.shape[0],y.shape[1]),
                                dtype=torch.int64,
                                device = device,
                                )
            classify_orig = list()
            classify_optim = list()
        def con_seq(confidence,y,return_optimization=0.2):
            result_seq =np.array([],dtype='int32')
            for i in range(y.shape[1]):
                y_ = y[:,i]
                confidence_ = confidence[:,i]
                result = confidence_[torch.arange(y_.shape[0]), y_].exp() >= return_optimization
                result_seq = np.concatenate((result_seq,result), axis=1)
            return  result_seq
        ################################################################
        #                    Attention optimisation                    #
        ################################################################

        # Loop over batches
        for batch, (X1_, y1_, label_) in enumerate(batches):
            print(batch,batch,batch)
            confidence_orig1 = torch.empty((y1_.shape[0], y1_.shape[1]),
                                           dtype=torch.int64,
                                           device=device,
                                           )
            confidence_optim1 = torch.empty((y1_.shape[0], y1_.shape[1]),
                                            dtype=torch.int64,
                                            device=device,
                                            )
            confidence1, attention1, class_out1 = self.predict1(X1_, y1_, steps=y1_.shape[1])

            # confidence = confidence.squeeze(1)  # 128*1*100-->128*100
            # attention = attention.squeeze(1)  # 128*1*10-->128*10
            # classify = class_out.squeeze(1)
            for step in range(1):
                confidence = confidence1[:,step,:].squeeze(1)
                attention = attention1[:,step,:].squeeze(1)
                classify = class_out1[:,step,:].squeeze(1)
                # Compute initial attention and confidence
                # from pudb import set_trace;set_trace()
                y_ = y1_[:,step]

                # import pdb;pdb.set_trace()
                # Count confidence >= 0.2 of non-optimized datapoints
                if return_optimization is not None:
                    result_orig = confidence[torch.arange(y_.shape[0]), y_].exp() >= return_optimization
                    confidence_orig1[:,step] = result_orig
                    classify_orig1=classify.detach().clone()
                # import pdb;pdb.set_trace()

                # Make attention variable
                attn = Variable(attention.detach().clone(), requires_grad=True)
                # import pdb;pdb.set_trace()
                # Set optimizer
                optimizer = optim.Adam([attn], lr=0.1)
                criterion = nn.NLLLoss()

                # Encode values of X
                with torch.no_grad():
                    X_, _ = self.encoder(X1_)

                # Perform iterations
                for iteration in range(int(iterations)):
                    # Clear optimizer
                    optimizer.zero_grad()

                    # Add decoding function
                    def decode(input, attn, softmax=False):
                        if softmax: attn = F.softmax(attn, dim=1)
                        return self.decoder_event(input, attn), self.decoder_class(input, attn),

                    # from pudb import set_trace;set_trace()
                    # Perform prediction
                    pred_confi, pred_class = decode(X_, attn, softmax=iteration > 0)
                    # import pdb;pdb.set_trace()

                    # Compute loss
                    loss = criterion(pred_confi, y_) + criterion(pred_class, label_)
                    # print(loss)

                    # Perform backpropagation
                    loss.backward()
                    optimizer.step()

                    # Update progress if necessary
                    if verbose: progress.update()

                # Perform final softmax
                if iterations > 0: attn = F.softmax(attn, dim=1)

                # Detach attention - memory optimization
                attn = attn.detach()

                # Get confidence levels
                confidence_,_ = self.decoder_event(X_, attn)
                # import pdb;pdb.set_trace()
                confidence_ = confidence_[torch.arange(y_.shape[0]), y_].exp().detach()
                # import pdb;pdb.set_trace()
                confidence = confidence[torch.arange(y_.shape[0]), y_].exp().detach()
                # import pdb;pdb.set_trace()

                # Check where confidence improved
                mask = confidence_ > confidence

                # Store attention if we improved
                attention[mask] = attn[mask]
                # import pdb;pdb.set_trace()

                # Recompute confidence
                with torch.no_grad():
                    confidence,_ = self.decoder_event(
                        X=X_,
                        attention=attention,
                    ).exp()
                    classify = self.decoder_class(
                        X=X_,
                        attention=attention,
                    )

                    # Count confidence >= 0.2 of optimized datapoints
                    if return_optimization is not None:
                        result_optim = confidence[torch.arange(y_.shape[0]), y_] >= return_optimization
                        confidence_optim1[:,step]  = result_optim
                        classify_optim1 = classify.detach().clone()
            confidence_optim[batch*batch_size:(batch+1)*batch_size] = confidence_optim1
            confidence_orig[batch*batch_size:(batch+1)*batch_size]  =confidence_orig1
            classify_optim.append(classify_optim1)
            classify_orig.append(classify_orig1)
                # Add confidence and attention to result
                # result_confidence.append(confidence.cpu())
                # result_attention.append(attention.cpu())



        # Combine confidence and attention into tensor
        # and cast to original device
        #     confidence = torch.cat(result_confidence).to(original_device)
        #     attention = torch.cat(result_attention).to(original_device)
        classify_orig = torch.cat(classify_orig)
        classify_optim = torch.cat(classify_optim)
        loss_orig = criterion(classify_orig, label)
        loss_optim = criterion(classify_optim, label)

        # Close progress if necessary
        if verbose: progress.close()
        self.save('../model/'+model_path)
        # Return confidence optimization if necessary
        if return_optimization is not None:
            confidence_orig = confidence_orig.numpy()
            confidence_optim = confidence_optim.numpy()
            # Return result
            return confidence_orig, confidence_optim, loss_orig, loss_optim

        # Return result
        return confidence, attention, inverse

    ########################################################################
    #                           Save/load model                            #
    ########################################################################

    def save(self, outfile):
        """Save model to output file.

            Parameters
            ----------
            outfile : string
                File to output model.
            """
        # Save to output file
        torch.save(self.state_dict(), outfile)

    @classmethod
    def load(cls, infile, device=None):
        """Load model from input file.

            Parameters
            ----------
            infile : string
                File from which to load model.
            """
        # Load state dictionary
        state_dict = torch.load(infile, map_location=device)

        # Get input variables from state_dict
        input_size    = state_dict.get('embedding.weight').shape[0]
        output_size   = state_dict.get('decoder_event.out.weight').shape[0]
        class_size   = state_dict.get('decoder_class.out.weight').shape[0]
        hidden_size   = state_dict.get('embedding.weight').shape[1]
        num_layers    = 1 # TODO
        max_length    = state_dict.get('decoder_attention.attn.weight').shape[0]
        bidirectional = state_dict.get('decoder_attention.attn.weight').shape[1] // hidden_size != num_layers
        LSTM          = False # TODO
        # import pdb;pdb.set_trace()
        # Create ContextBuilder
        result = cls(
            input_size    = input_size,
            output_size   = output_size,
            class_size    = class_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            max_length    = max_length,
            bidirectional = bidirectional,
            LSTM          = LSTM,
        )

        # Cast to device if necessary
        if device is not None: result = result.to(device)

        # Set trained parameters
        result.load_state_dict(state_dict)

        # Return result
        return result
