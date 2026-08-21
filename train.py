import os
os.environ["DATA_FOLDER"] = "./"

import argparse
import random

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

import pandas as pd
from imc_cellannot_utils import load_data, load_cell_ontology, compute_R, plot_ontology, propagate_labels


def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
    R_batch = R.expand(len(x),R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
    return final_out


class ConstrainedFFNNModel(nn.Module):
    """ C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss """
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R):
        super(ConstrainedFFNNModel, self).__init__()

        self.nb_layers = hyperparams['num_layers']
        self.R = R

        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers-1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)

        self.drop = nn.Dropout(hyperparams['dropout'])


        self.sigmoid = nn.Sigmoid()
        if hyperparams['non_lin'] == 'tanh':
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()

    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers-1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='Train neural network')

    # Required  parameters
    parser.add_argument('--X', type=str, required=True,
                        help='path to feature matrix (.npy, .csv, or .parquet)')
    parser.add_argument('--Y', type=str, required=True,
                        help='path to label matrix (.csv, or .parquet); .csv/.parquet column names are used as node names for --rm_root_nodes matching')
    parser.add_argument('--ontology', type=str, required=True,
                        help='path to cl.obo (Cell Ontology in OBO format)')
    parser.add_argument('--extra_nodes', type=str, nargs='*', default=None,
                        help='non-ontology labels to add as isolated nodes with no hierarchy constraints (e.g. empty unknown)')
    parser.add_argument('--run_name', type=str, required=True,
                        help='name for this run, used as the logs subfolder')
    
    # hyper parameters
    parser.add_argument('--batch_size', type=int, required=True,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, required=True,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, required=True,
                        help='dropout probability')
    parser.add_argument('--hidden_dim', type=int, required=True,
                        help='size of the hidden layers')
    parser.add_argument('--num_layers', type=int, required=True,
                        help='number of hidden layers')
    parser.add_argument('--weight_decay', type=float, required=True,
                        help='weight decay')
    parser.add_argument('--non_lin', type=str, required=True,
                        help='non linearity function to be used in the hidden layers')

    # Other parameters
    parser.add_argument('--rm_root_nodes', type=str, nargs='*', default=[],
                    help='node names to exclude from loss and scoring (e.g. root GO0003674); requires Y in .csv or .parquet format')
    parser.add_argument('--val_ratio', type=float, default=0.3,
                        help='fraction of the data held out for validation (default: 0.3)')
    parser.add_argument('--device', type=int, default=0,
                        help='device (default:0)')
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help='Max number of epochs to train (default:2000)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default:0)')
    parser.add_argument('--debug', action='store_true',
                        help='write debug outputs (e.g. ontology plot) to the logs folder')

    args = parser.parse_args()
    hyperparams = {'batch_size':args.batch_size, 'num_layers':args.num_layers, 'dropout':args.dropout, 'non_lin':args.non_lin, 'hidden_dim':args.hidden_dim, 'lr':args.lr, 'weight_decay':args.weight_decay}

    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Pick device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    # Load X
    X = load_data(args.X)
    
    # Load Y
    Y = load_data(args.Y)
    if not isinstance(Y, pd.DataFrame):
        raise ValueError(f"Y must be a .csv or .parquet file with column names — got {args.Y!r}")

    # Load ontology (pruned to Y columns + all their ancestors)
    initial_nodes = list(Y.columns)
    extra_nodes = set(args.extra_nodes) if args.extra_nodes else set()
    ontology_nodes = [n for n in initial_nodes if n not in extra_nodes]
    g = load_cell_ontology(args.ontology, nodes=ontology_nodes, extra_nodes=args.extra_nodes)

    if args.debug:
        plot_ontology(g, output_path=f"logs/{args.run_name}/ontology.html")

    # Include all ancestor nodes of Y and propoagate positive labels updward (e.g. CD8+ T cell → T cell = 1, lymphocyte = 1, leukocyte = 1, ...)
    Y = propagate_labels(Y, g)

    # Remove requested root nodes from loss and scoring
    if args.rm_root_nodes:
        missing = set(args.rm_root_nodes) - set(Y.columns)
        if missing:
            raise ValueError(f"--rm_root_nodes contains names not found in Y columns: {sorted(missing)}")
    to_eval = ~Y.columns.isin(args.rm_root_nodes)
    
    # Y data management for input
    Y = Y.values                # only Y values for model input
    nodes = list(Y.columns)     # use expanded nodes for hierarchy

    # Build R from expanded full node set
    R = torch.tensor(compute_R(g, nodes)).unsqueeze(0).to(device)                               # R_ij = 1 if i is an ancestor of j. Shape (1, n(nodes), n(nodes)) for broadcasting to batch dimension


    # Train/val split (after label propagation so propagated labels are consistent across splits)
    train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=args.val_ratio, random_state=seed)

    # scale and send to tensors to device (e.g gpu)
    scaler = preprocessing.StandardScaler().fit(train_X)                                        # z-score scale X
    train_X = torch.tensor(scaler.transform(train_X)).to(device)
    val_X = torch.tensor(scaler.transform(val_X)).to(device)
    train_Y = torch.tensor(train_Y).to(device)
    val_Y = torch.tensor(val_Y).to(device)

    # Create loaders
    train_dataset = [(x, y) for (x, y) in zip(train_X, train_Y)]
    val_dataset = [(x, y) for (x, y) in zip(val_X, val_Y)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)


    num_epochs = args.num_epochs

    # Create the model
    model = ConstrainedFFNNModel(X.shape[1], args.hidden_dim, Y.shape[1], hyperparams, R)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    # Set patience
    patience, max_patience = 20, 20
    max_score = 0.0

    # Create folder for this run (if it does not exist)
    if not os.path.exists('logs/'+str(args.run_name)+'/'):
         os.makedirs('logs/'+str(args.run_name)+'/')

    for epoch in range(num_epochs):
        total_train = 0.0
        correct_train = 0.0
        model.train()

        train_score = 0

        for i, (x, labels) in enumerate(train_loader):

            x = x.to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            output = model(x.float())

            # MCLoss
            constr_output = get_constr_out(output, R)
            train_output = labels*output.double()
            train_output = get_constr_out(train_output, R)
            train_output = (1-labels)*constr_output.double() + labels*train_output
            loss = criterion(train_output[:, to_eval], labels[:, to_eval])
            predicted = constr_output.data > 0.5
            # Total number of labels
            total_train += labels.size(0) * labels.size(1)
            # Total correct predictions
            correct_train += (predicted == labels.byte()).sum()

            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

        model.eval()
        constr_output = constr_output.to('cpu')
        labels = labels.to('cpu')
        train_score = average_precision_score(labels[:, to_eval], constr_output.data[:, to_eval], average='micro')

        for i, (x,y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            constrained_output = model(x.float())
            predicted = constrained_output.data > 0.5
            # Total number of labels
            total = y.size(0) * y.size(1)
            # Total correct predictions
            correct = (predicted == y.byte()).sum()

            #Move output and label back to cpu to be processed by sklearn
            cpu_constrained_output = constrained_output.to('cpu')
            y = y.to('cpu')

            if i == 0:
                constr_val = cpu_constrained_output
                y_val = y
            else:
                constr_val = torch.cat((constr_val, cpu_constrained_output), dim=0)
                y_val = torch.cat((y_val, y), dim =0)

        score = average_precision_score(y_val[:, to_eval], constr_val.data[:, to_eval], average='micro')

        if score >= max_score:
            patience = max_patience
            max_score = score
        else:
            patience = patience - 1

        floss= open('logs/'+str(args.run_name)+'/measures_batch_size_'+str(args.batch_size)+'_lr_'+str(args.lr)+'_weight_decay_'+str(args.weight_decay)+'_seed_'+str(args.seed)+'_num_layers_'+str(args.num_layers)+'._hidden_dim_'+str(args.hidden_dim)+'_dropout_'+str(args.dropout)+'_'+args.non_lin, 'a')
        floss.write('\nEpoch: {} - Loss: {:.4f}, Accuracy train: {:.5f}, Accuracy: {:.5f}, Precision score: ({:.5f})\n'.format(epoch,
                    loss, float(correct_train)/float(total_train), float(correct)/float(total), score))
        floss.close()

        if patience == 0:
            break

if __name__ == "__main__":
    main()
