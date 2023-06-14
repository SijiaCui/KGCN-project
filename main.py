import time
import pandas as pd
import numpy as np
import argparse
import random
from model import KGCN, KGCNDataset
from data_loader import DataLoader
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def parse_args():
    # prepare arguments (hyperparameters)
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')

    args = parser.parse_args()
    print(args, flush=True)
    return args

def train_test_loader(args, df_dataset):
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=1 - args.ratio, shuffle=False, random_state=999)
    train_dataset = KGCNDataset(x_train)
    test_dataset = KGCNDataset(x_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    return train_loader, test_loader


def training(args):
    data_loader = DataLoader(args.dataset)
    train_loader, test_loader = train_test_loader(args, data_loader.load_dataset())
    
    # prepare network, loss function, optimizer
    num_user, num_entity, num_relation = data_loader.get_num()
    user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    net = KGCN(num_user, num_entity, num_relation, data_loader.load_kg(), args, device).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)

    # train
    loss_list = []
    test_loss_list = []
    auc_score_list = []

    start = time.time()
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(user_ids, item_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()
        
        # evaluate per every epoch
        with torch.no_grad():
            test_loss = 0
            total_roc = 0
            for user_ids, item_ids, labels in test_loader:
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs = net(user_ids, item_ids)
                test_loss += criterion(outputs, labels).item()
                total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            
            print('[Epoch {}] train_loss:{} test_loss:{} acc:{}'.format(
                epoch+1, running_loss / len(train_loader), 
                test_loss / len(test_loader), 
                total_roc / len(test_loader)), flush=True)
            loss_list.append(running_loss / len(train_loader))
            test_loss_list.append(test_loss / len(test_loader))
            auc_score_list.append(total_roc / len(test_loader))
    
    print('training time:{}'.format(time.time()-start))
    return loss_list, test_loss_list, auc_score_list


if __name__=="__main__":
    args = parse_args()
    training(args)
