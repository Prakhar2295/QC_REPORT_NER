import pandas as pd
import matplotlib.pyplot as plt
import logging
import re
import json
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import trange
from tqdm import tqdm_notebook as tqdm
from transformers import BertForTokenClassification, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
from data_preparation import data_preparation,ReportDataset
from data_preprocessing import data_preprocessing


def data_loader():
    data = data_preprocessing.remove_white_spaces_from_entities()

    total = len(data)
    train_data, val_data = data[:400], data[400:]

    label_list = ["CERTIFICATE_NO","ISSUED_DATE","ORDER_NO","PURCHASE_ORDER","SUPPLIER_NAME","CUSTOMER_NAME","PLATE_NO","O","MISC"]

    label2id = {v:k for k,v in enumerate(label_list)}

    TOKENIZER = BertTokenizerFast("D:/FSDS/NLP/Report_NER/Bert/vocab.txt",do_lower_case = True)

    train_d = ReportDataset(train_data,TOKENIZER,label2id,512,is_test=False)
    val_d = ReportDataset(val_data,TOKENIZER,label2id,512,is_test=False)

    train_sampler = RandomSampler(train_d)
    train_dl = DataLoader(train_d, sampler=train_sampler, batch_size=8,drop_last=True)

    val_dl = DataLoader(val_d, batch_size=8,drop_last=True)

    return train_dl,val_dl


def get_special_tokens(tokenizer, tag2idx):
    vocab = tokenizer.get_vocab()
    pad_tok = vocab["[PAD]"]
    sep_tok = vocab["[SEP]"]
    cls_tok = vocab["[CLS]"]
    o_lab = tag2idx["O"]

    return pad_tok, sep_tok, cls_tok, o_lab


def annot_confusion_matrix(valid_tags, pred_tags):

    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn's `confusion_matrix`.
    """

    # Create header from unique tags
    header = sorted(list(set(valid_tags + pred_tags)))

    # Calculate the actual confusion matrix
    matrix = confusion_matrix(valid_tags, pred_tags, labels=header)

    # Final formatting touches for the string output
    mat_formatted = [header[i] + "\t\t\t" + str(row) for i, row in enumerate(matrix)]
                #p =[header[i] + "\t\t\t" + str(row) for i,row in enumerate(matrix)]
    content = "\t" + " ".join(header) + "\n" + "\n".join(mat_formatted)

    return content

def flat_accuracy(valid_tags, pred_tags):
    return (np.array(valid_tags) == np.array(pred_tags)).mean()





