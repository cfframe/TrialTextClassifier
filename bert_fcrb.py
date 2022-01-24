import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer

from src.bert_serums_dataset import Dataset
from src.bert_model import BertClassifier


def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=5, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=5)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    results = []

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            # CFF train_label = train_label.to(torch.int64)
            train_label = train_label.to(torch.int64)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                # CFF train_label = train_label.to(torch.int64)
                val_label = val_label.to(torch.int64)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        results.append([epoch_num + 1,
                        total_loss_train / len(train_data), total_acc_train / len(train_data),
                        total_loss_val / len(val_data), total_acc_val / len(val_data)])
        print(
            f'\nEpochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

    #results_df = pd.DataFrame(data=results, columns=['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])

    #return results_df


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=5)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


def main():

    datapath = '.data/serums_fcrb.csv'
    df = pd.read_csv(datapath)
    df.head()

    df.groupby(['category']).size().plot.bar()
    plt.show()

    # Preprocessing data
    # Split df into training, validation, test at 80:10:10
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])

    print(len(df_train), len(df_val), len(df_test))

    # Training loop

    EPOCHS = 2
    model = BertClassifier()
    LR = 1e-6

    # df = train(model, df_train, df_val, LR, EPOCHS)
    train(model, df_train, df_val, LR, EPOCHS)

    # x = df['Epoch'].to_list()
    # trn_acc_label = 'Train Accuracy'
    # trn_acc = df[trn_acc_label].to_list()
    # val_acc_label = 'Val Accuracy'
    # val_acc = df[val_acc_label].to_list()
    #
    # plt.plot(x, trn_acc, label=trn_acc_label)
    # plt.plot(x, val_acc, label=val_acc_label)
    #
    # plt.legend()
    # plt.show()

    evaluate(model, df_test)


if __name__ == '__main__':
    main()
