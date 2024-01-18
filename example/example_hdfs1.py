# Other imports
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()  # 或者 StandardScaler()
import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# DeepCASE Imports
import sys
sys.path.append("../")
from deepcase.preprocessing import Preprocessor
from deepcase.context_builder import ContextBuilder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import pandas as pd
import seaborn as sns

sns.set()
if __name__ == "__main__":

    train_x = np.load('../../DeepCASE_bid/DeepCASE/data/MQTT/train_x_lbs33_clean1.npy')
    test_x = np.load('../../DeepCASE_bid/DeepCASE/data/MQTT/test_x_lbs33_clean1.npy')
    val_x = np.load('../../DeepCASE_bid/DeepCASE/data/MQTT/val_x_lbs33_clean1.npy')
    label_val = np.load('../../DeepCASE_bid/DeepCASE/data/MQTT/val_y_lbs33_clean1.npy')
    label_train = np.load('../../DeepCASE_bid/DeepCASE/data/MQTT/train_y_lbs33_clean1.npy')
    label_test = np.load('../../DeepCASE_bid/DeepCASE/data/MQTT/test_y_lbs33_clean1.npy')

    pkt_len = 20
    context_train = train_x[:, :pkt_len]
    context_test = test_x[:, :pkt_len]
    context_val = val_x[:, :pkt_len]
    # events_train = train_x[:, pkt_len:pkt_len+10]
    # events_test = test_x[:, pkt_len:pkt_len+10]
    # events_val = val_x[:, pkt_len:pkt_len+10]
    events_train = train_x[:, pkt_len]
    events_test = test_x[:, pkt_len]
    events_val = val_x[:, pkt_len]


    ########################################################################
    #                       Training ContextBuilder                        #
    ########################################################################

    # Create ContextBuilder
    context_builder = ContextBuilder(
        input_size=30,  # Number of input features to expect
        # embedded_size = 16,
        output_size=30,  # Same as input size
        class_size=5,
        hidden_size=128,  # Number of nodes in hidden layer, in paper we set this to 128
        max_length=pkt_len,  # Length of the context, should be same as context in Preprocessor
    )

    # Cast to cuda if available
    if torch.cuda.is_available():
        context_builder = context_builder.to('cuda')
    import pdb;pdb.set_trace()
    # Train the ContextBuilder
    context_builder = ContextBuilder.load('../model/IDS2012/lbs10_1_ep100_7.pth')

    context_builder.fit(
        X_train=context_train,
        X_val=context_val,
        y_train=events_train.reshape(-1, 10),
        y_val=events_val.reshape(-1, 10),
        label_train=label_train.reshape(-1, 1),
        label_val=label_val.reshape(-1, 1),
        model_path='IDS2012/lbs'+str(pkt_len)+'_1_tt10_7.pth',
        epochs=100,  # Number of epochs to train with
        batch_size=128,
        learning_rate=0.01,
        verbose=True,  # If True, prints progress
    )
    # import pdb;pdb.set_trace()
    context_builder = ContextBuilder.load('../model/MQTT/lbs20_1_ep100_3.pth')
    # import pdb;pdb.set_trace()
    # model1 = copy.deepcopy(context_builder)
    confidence, attention, confidence_orig, confidence_optim,loss_orig, loss_optim = context_builder.query(
        X=context_train,
        y=events_train.reshape(-1, 1),
        label=label_train.reshape(-1, 1),
        model_path='IDS2012/lbs20_1_ep100_4_query.pth',
        iterations=200,
        batch_size=2048,
        return_optimization=0.2,
        verbose=True,
    )
    # print(confidence_orig.sum(), confidence_optim.sum(),loss_orig, loss_optim)
    # # # import pdb;pdb.set_trace()
    context_builder = ContextBuilder.load('../model/IDS2012/query/lbs' + str(pkt_len) + '_1_noquery.pth')
    # context_builder1 = ContextBuilder.load('../model/IDS2012/query/lbs'+str(pkt_len)+'_1_query08.pth')


    confidence, attention, class_out,attn_x = context_builder.predict(
        X=torch.from_numpy(context_test),
        # steps= events_train.shape[1],
    )

    y_test = label_test
    y_pred = class_out
    len_pred = confidence
    len_test = events_test


    def report_results_pktlen(y_true, y_pred):
        accuracy, precision, recall, f1 = 0, 0, 0, 0
        for i in range(y_true.shape[1]):
            Y_true = y_true[:, i]
            Y_pred = y_pred[:, i]
            accuracy += accuracy_score(Y_true, Y_pred)
            f1 += f1_score(Y_true, Y_pred, average='weighted')
            precision += precision_score(Y_true, Y_pred, average='weighted')
            recall += recall_score(Y_true, Y_pred, average='weighted')

        test_string_pre = '{:05.4f}'.format(accuracy / float(y_true.shape[1])) + \
                          " " + '{:05.4f}'.format(precision / float(y_true.shape[1])) + \
                          " " + '{:05.4f}'.format(recall / float(y_true.shape[1])) + " " + '{:05.4f}'.format(
            f1 / float(y_true.shape[1])) + "\n"

        output_header = "ACC   Pre    Recall    F1\n"
        output_string = test_string_pre
        print(output_header)
        print(output_string)

    print('The result of packet length prediction:')
    # report_results_pktlen(len_test, len_pred)
    print('accuracy_score', accuracy_score(len_test, len_pred))
    print(classification_report(
        y_true=len_test,
        y_pred=len_pred,
        digits=4,
    ))
    print('The result of flow lable prediction:')
    print(classification_report(
        y_true=y_test,
        y_pred=y_pred,
        digits=4,
    ))

