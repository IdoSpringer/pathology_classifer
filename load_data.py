import torch
from torch.utils.data import Dataset, DataLoader
import csv
from TCR_Autoencoder.tcr_autoencoder import PaddingAutoencoder
from random import shuffle
import numpy as np

# todo: include v and j genes

class McPAS_TCR(Dataset):
    def __init__(self, csv_file, tcr_atox, params, label_type):
        self.max_tcr_len = params['max_len']
        self.ae_file = params['ae_file']
        self.enc_dim = params['enc_dim']
        self.batch_size = params['batch_size']
        self.class_limit = params['out_dim']
        tcrs = []
        labels = []
        with open(csv_file, 'r', encoding='unicode_escape') as file:
            file.readline()
            reader = csv.reader(file)
            index = 0
            for line in reader:
                index += 1
                tcr, pathology = line[1], line[4]
                if label_type == 'pathology':
                    label = line[4]
                elif label_type == 'category':
                    label = line[3]
                elif label_type == 'protein':
                    label = line[9]
                if tcr == 'NA' or label == 'NA':
                    continue
                if len(tcr) > self.max_tcr_len:
                    continue
                if any(key in tcr for key in ['#', '*', 'b', 'f', 'y', '~', 'O']):
                    continue
                tcrs.append(tcr)
                labels.append(label)
        # The index number of a label will be its relative frequency
        # (0 - the most frequent, last - the most rare)
        l_set = set(labels)
        count_dict = {l: labels.count(l) for l in l_set}
        count_dict = sorted(count_dict, key=lambda k: count_dict[k], reverse=True)
        label_dict = {l: index for index, l in enumerate(count_dict)}
        for i in range(len(labels)):
            labels[i] = label_dict[labels[i]]
        pruned_tcrs = []
        pruned_labels = []
        for tcr, label in zip(tcrs, labels):
            if label < self.class_limit:
                pruned_tcrs.append(tcr)
                pruned_labels.append(label)
        self.tcrs = pruned_tcrs
        self.labels = pruned_labels
        self.tcr_amino_to_ix = tcr_atox
        self.params = params

    def __len__(self):
        return len(self.tcrs)

    def __getitem__(self, item):
        return self.tcrs[item]

    def convert_data(self, tcrs):
        for i in range(len(tcrs)):
            tcrs[i] = self.pad_tcr(tcrs[i])

    def pad_tcr(self, tcr):
        padding = torch.zeros(self.max_tcr_len, 20 + 1)
        tcr = tcr + 'X'
        for i in range(len(tcr)):
            amino = tcr[i]
            padding[i][self.tcr_amino_to_ix[amino]] = 1
        return padding

    def get_batches_for_ae(self, tcrs, labels):
        """
        Get batches from the data
        """
        # Load autoencoder
        autoencoder = PaddingAutoencoder(self.max_tcr_len, 21, self.enc_dim)
        checkpoint = torch.load(self.ae_file)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        for param in autoencoder.parameters():
            param.requires_grad = False
        autoencoder.eval()
        # Shuffle
        z = list(zip(tcrs, labels))
        shuffle(z)
        tcrs, pathologies = zip(*z)
        tcrs = list(tcrs)
        labels = list(labels)
        # Initialization
        batches = []
        index = 0
        self.convert_data(tcrs)
        # Go over all data
        while index < len(tcrs) // self.batch_size * self.batch_size:
            # Get batch sequences and math tags
            # Add batch to list
            batch_tcrs = tcrs[index:index + self.batch_size]
            tcr_tensor = torch.zeros((self.batch_size, self.max_tcr_len, 21))
            for i in range(self.batch_size):
                tcr_tensor[i] = batch_tcrs[i]
            concat = tcr_tensor.view(self.batch_size, self.max_tcr_len * 21)
            encoded_tcrs = autoencoder.encoder(concat)
            batch_labels = labels[index:index + self.batch_size]
            batches.append((encoded_tcrs, batch_labels))
            # Update index
            index += self.batch_size
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

    def get_encodings(self, tcrs, labels):
        batches = self.get_batches_for_ae(tcrs, labels)
        tcrs = []
        labels = []
        for batch in batches:
            encoded_tcrs, batch_labels = batch
            for tcr in encoded_tcrs:
                tcrs.append(tcr.numpy())
            labels.extend(batch_labels)
        tcrs = np.array(tcrs)
        return tcrs, labels

    def encode_data(self):
        self.tcrs, self.labels = self.get_encodings(self.tcrs, self.labels)

    pass