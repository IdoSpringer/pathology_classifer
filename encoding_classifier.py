import sys
import argparse
import torch
import sklearn
import autosklearn.classification
from sklearn.model_selection import train_test_split
from load_data import McPAS_TCR
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE


def train(train_tcrs, test_tcrs, train_labels, test_labels, args):
    smote = args.over_sampling
    if smote:
        print('using smote')
        smt = SMOTE()
        train_tcrs, train_labels = smt.fit_sample(train_tcrs, train_labels)
    if args.model_type == 'knn':
        neigh = KNeighborsClassifier(n_neighbors=args.n_neighbors if args.n_neighbors else 5)
        neigh.fit(train_tcrs, train_labels)
        y_hat_train = neigh.predict(train_tcrs)
        y_hat_test = neigh.predict(test_tcrs)
        pass
    elif args.model_type == 'auto_ml':
        automl = autosklearn.classification.AutoSklearnClassifier()
        automl.fit(train_tcrs, train_labels)
        y_hat_train = automl.predict(train_tcrs)
        y_hat_test = automl.predict(test_tcrs)
    elif args.model_type == 'mlp':
        pass
    print("Train accuracy:", sklearn.metrics.accuracy_score(train_labels, y_hat_train))
    print("Test accuracy:", sklearn.metrics.accuracy_score(test_labels, y_hat_test))


def main(args):
    # Data
    csv_file = 'McPAS-TCR.csv'
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
    # Set all parameters and program arguments
    params = {}
    params['ae_file'] = 'TCR_Autoencoder/tcr_autoencoder.pt'
    params['emb_dim'] = 10
    params['enc_dim'] = 30
    params['out_dim'] = args.n_classes
    # Load autoencoder params
    checkpoint = torch.load(params['ae_file'])
    params['max_len'] = checkpoint['max_len']
    params['batch_size'] = checkpoint['batch_size']
    # Load data
    data = McPAS_TCR(csv_file, tcr_atox, params, label_type=args.label_type)
    data.encode_data()
    train_tcrs, test_tcrs, train_labels, test_labels = train_test_split(data.tcrs, data.labels, test_size=0.2)
    # Train the model
    train(train_tcrs, test_tcrs, train_labels, test_labels, args)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label_type')
    parser.add_argument('n_classes', type=int)
    parser.add_argument('model_type')
    parser.add_argument('--n_neighbors', type=int)
    parser.add_argument('--over_sampling', type=bool)
    args = parser.parse_args()
    main(args)
