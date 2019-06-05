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
from imblearn.over_sampling import SMOTE
from TCR_Autoencoder.tcr_autoencoder import PaddingAutoencoder
from models import MLP


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
        if pathology < class_limit:
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

'''
def encode_tcrs(tcrs, params, args)
    max_length = params['max_len']
    batch_size = params['batch_size']
    # Load autoencoder
    autoencoder = PaddingAutoencoder(max_length, 21, params['enc_dim'])
    checkpoint = torch.load(args['ae_file'])
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.eval()
    pass
'''


def get_batches(tcrs, tcr_atox, pathologies, params, args):
    """
    Get batches from the data
    """
    max_length = params['max_len']
    batch_size = params['batch_size']
    # Load autoencoder
    autoencoder = PaddingAutoencoder(max_length, 21, params['enc_dim'])
    checkpoint = torch.load(args['ae_file'])
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.eval()
    # Shuffle
    z = list(zip(tcrs, pathologies))
    shuffle(z)
    tcrs, pathologies = zip(*z)
    tcrs = list(tcrs)
    pathologies = list(pathologies)
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
        concat = tcr_tensor.view(batch_size, max_length * 21)
        encoded_tcrs = autoencoder.encoder(concat)
        batch_pathologies = pathologies[index:index + batch_size]
        batches.append((encoded_tcrs, batch_pathologies))
        # Update index
        index += batch_size
    '''
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
    '''
    # Return list of all batches
    return batches


def read_batches(batches):
    tcrs = []
    pathologies = []
    for batch in batches:
        encoded_tcrs, batch_pathologies = batch
        for tcr in encoded_tcrs:
            tcrs.append(tcr.numpy())
        pathologies.extend(batch_pathologies)
    tcrs = np.array(tcrs)
    return tcrs, pathologies


def train_epoch(train_tcrs, train_pathologies, model, loss_function, optimizer, device):
    model.train()
    z = list(zip(train_tcrs, train_pathologies))
    shuffle(z)
    train_tcrs, train_pathologies = zip(*z)
    '''
    index = 0
    batches = []
    while index < len(train_tcrs):
        batches.append((train_tcrs[index:index+]))
    '''
    total_loss = 0
    for tcr, pathology in zip(train_tcrs, train_pathologies):
        # Move to GPU
        tcr = torch.tensor(tcr).to(device)
        pathology = torch.tensor(pathology).view(1).to(device)
        model.zero_grad()
        probs = model(tcr).view(1, -1)
        # Compute loss
        loss = loss_function(probs, pathology)
        # print(loss.item())
        # Update model weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # Return average loss
    return total_loss / len(train_tcrs)


def train_model(train_tcrs, train_pathologies, test_tcrs, test_pathologies,
                device, args, params):
    """
    Train and evaluate the model
    """
    losses = []
    # We use Cross-Entropy loss
    loss_function = nn.CrossEntropyLoss()
    # Set model with relevant parameters
    model = MLP(params['enc_dim'], params['out_dim'], device)
    # Move to GPU
    model.to(device)
    # We use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    # Train several epochs
    for epoch in range(params['epochs']):
        print('epoch:', epoch + 1)
        # Train model and get loss
        loss = train_epoch(train_tcrs, train_pathologies, model, loss_function, optimizer, device)
        losses.append(loss)
        # evaluate
        acc = evaluate(model, train_tcrs, train_pathologies, device)
        print('train accuracy:', acc)
        acc = evaluate(model, test_tcrs, test_pathologies, device)
        print('test accuracy:', acc)
    return model


def evaluate(model, tcrs, pathologies, device):
    model.eval()
    accuracy = 0
    samples = 0
    for tcr, pathology in zip(tcrs, pathologies):
        # Move to GPU
        tcr = torch.tensor(tcr).to(device)
        probs = model(tcr)
        pred = torch.argmax(probs)
        print(pred.item())
        if pred.item() == pathology:
            accuracy += 1
    accuracy /= len(tcrs)
    return accuracy


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
    params['epochs'] = 200
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
    train_tcrs, test_tcrs, train_pathologies, test_pathologies = train_test_split(tcrs, pathologies, test_size=0.2)
    # train with smote
    train_batches = get_batches(train_tcrs, tcr_atox, train_pathologies, params, args)
    train_tcrs, train_pathologies = read_batches(train_batches)
    smt = SMOTE()
    train_tcrs, train_pathologies = smt.fit_sample(train_tcrs, train_pathologies)
    # test
    test_batches = get_batches(test_tcrs, tcr_atox, test_pathologies, params, args)
    test_tcrs, test_pathologies = read_batches(test_batches)
    # Train the model
    model = train_model(train_tcrs, train_pathologies, test_tcrs, test_pathologies,
                        device, args, params)
    pass


if __name__ == '__main__':
    main(sys.argv)
