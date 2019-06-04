import csv
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from random import shuffle
import time
import numpy as np
import torch.autograd as autograd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from model import PathologyClassifier


m = 'McPAS-TCR.csv'


def get_lists_from_pairs(csv_file, max_len, class_limit):
    tcrs = []
    pathologies = []
    with open(csv_file, 'r', encoding='unicode_escape') as file:
        file.readline()
        reader = csv.reader(file)
        index = 0
        for line in reader:
            index += 1
            tcr, pathology = line[1], line[4]
            if tcr == 'NA' or pathology == 'NA':
                continue
            if len(tcr) > max_len:
                continue
            if any(key in tcr for key in ['#', '*', 'b', 'f', 'y', '~', 'O']):
                continue
            tcrs.append(tcr)
            pathologies.append(pathology)
    # The index number of a pathology will be its relative frequency
    # (0 - the most frequent, last - the most rare)
    p_set = set(pathologies)
    count_dict = {p: pathologies.count(p) for p in p_set}
    count_dict = sorted(count_dict, key=lambda k: count_dict[k], reverse=True)
    pathology_dict = {p: index for index, p in enumerate(count_dict)}
    for i in range(len(pathologies)):
        pathologies[i] = pathology_dict[pathologies[i]]
    pruned_tcrs = []
    pruned_pathologies = []
    for tcr, pathology in zip(tcrs, pathologies):
        if pathology <= class_limit:
            pruned_tcrs.append(tcr)
            pruned_pathologies.append(pathology)
    return pruned_tcrs, pruned_pathologies


def convert_data(tcrs, tcr_atox, max_len):
    for i in range(len(tcrs)):
        tcrs[i] = pad_tcr(tcrs[i], tcr_atox, max_len)


def pad_tcr(tcr, amino_to_ix, max_length):
    padding = torch.zeros(max_length, 20 + 1)
    tcr = tcr + 'X'
    for i in range(len(tcr)):
        amino = tcr[i]
        padding[i][amino_to_ix[amino]] = 1
    return padding


def get_batches(tcrs, tcr_atox, pathologies, batch_size, max_length, class_limit):
    """
    Get batches from the data
    """
    # Initialization
    batches = []
    index = 0
    convert_data(tcrs, tcr_atox, max_length)
    # Go over all data
    while index < len(tcrs) // batch_size * batch_size:
        # Get batch sequences and math tags
        # Add batch to list
        batch_tcrs = tcrs[index:index + batch_size]
        tcr_tensor = torch.zeros((batch_size, max_length, 21))
        for i in range(batch_size):
            tcr_tensor[i] = batch_tcrs[i]
        batch_pathologies = pathologies[index:index + batch_size]
        batches.append((tcr_tensor, batch_pathologies))
        # Update index
        index += batch_size
    # pad data in last batch
    missing = batch_size - len(tcrs) + index
    padding_tcrs = ['X'] * missing
    padding_pathologies = [class_limit] * missing
    convert_data(padding_tcrs, tcr_atox, max_length)
    batch_tcrs = tcrs[index:] + padding_tcrs
    tcr_tensor = torch.zeros((batch_size, max_length, 21))
    for i in range(batch_size):
        tcr_tensor[i] = batch_tcrs[i]
    batch_pathologies = pathologies[index:] + padding_pathologies
    batches.append((tcr_tensor, batch_pathologies))
    # Update index
    index += batch_size
    # Return list of all batches
    return batches


def train_epoch(batches, model, loss_function, optimizer, device):
    model.train()
    shuffle(batches)
    total_loss = 0
    for batch in batches:
        tcrs, pathologies = batch
        # Move to GPU
        tcrs = tcrs.to(device)
        pathologies = torch.LongTensor(pathologies).to(device)
        model.zero_grad()
        probs = model(tcrs)
        # Compute loss
        loss = loss_function(probs, pathologies)
        # Update model weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # Return average loss
    return total_loss / len(batches)


def train_model(batches, test_batches, device, args, params):
    """
    Train and evaluate the model
    """
    losses = []
    # We use Cross-Entropy loss
    loss_function = nn.CrossEntropyLoss()
    # Set model with relevant parameters
    model = PathologyClassifier(params['emb_dim'], device, params['max_len'], 21, params['enc_dim'],
                                params['out_dim'] + 1, params['batch_size'], args['ae_file'], train_ae=True)
    # Move to GPU
    model.to(device)
    # We use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    # Train several epochs
    for epoch in range(params['epochs']):
        print('epoch:', epoch + 1)
        # Train model and get loss
        loss = train_epoch(batches, model, loss_function, optimizer, device)
        losses.append(loss)
        # evaluate
        acc = evaluate(model, batches, device)
        print('train accuracy:', acc)
        acc = evaluate(model, test_batches, device)
        print('test accuracy:', acc)

    return model


def evaluate(model, batches, device):
    model.eval()
    accuracy = 0
    samples = 0
    shuffle(batches)
    for batch in batches:
        tcrs, pathologies = batch
        # Move to GPU
        tcrs = torch.tensor(tcrs).to(device)
        pathologies = torch.LongTensor(pathologies).to(device)
        probs = model(tcrs)
        pred = torch.argmax(probs, dim=1)
        hits = pred == pathologies
        samples += len(hits)
        accuracy += int(sum(hits))
    # Return accuracy
    return accuracy / samples


def main(argv):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}

    # Set all parameters and program arguments
    device = 'cuda:1'
    args = {}
    # args['train_auc_file'] = argv[4]
    # args['test_auc_file'] = argv[5]
    args['ae_file'] = 'TCR_Autoencoder/tcr_autoencoder.pt'
    params = {}
    params['lr'] = 5 * 1e-4
    params['wd'] = 0
    params['epochs'] = 100
    params['emb_dim'] = 10
    params['enc_dim'] = 30
    class_limit = params['out_dim'] = 10
    params['dropout'] = 0.05
    params['train_ae'] = True
    # Load autoencoder params
    checkpoint = torch.load(args['ae_file'])
    params['max_len'] = checkpoint['max_len']
    params['batch_size'] = checkpoint['batch_size']

    # Load data
    csv_file = 'McPAS-TCR.csv'
    tcrs, pathologies = get_lists_from_pairs(csv_file, params['max_len'], class_limit)
    train_tcrs, test_tcrs, train_pathologies, test_pathologies = train_test_split(tcrs, pathologies, test_size = 0.2)
    # train
    train_batches = get_batches(train_tcrs, tcr_atox, train_pathologies, params['batch_size'], params['max_len'], class_limit)
    # test
    test_batches = get_batches(test_tcrs, tcr_atox, test_pathologies, params['batch_size'], params['max_len'], class_limit)

    # Train the model
    model = train_model(train_batches, test_batches, device, args, params)
    pass


if __name__ == '__main__':
    main(sys.argv)
