import torch 
import torch.nn as nn
import random, math, tqdm, argparse

from models import Conceptual_Similarity

torch.manual_seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)


def read_training_data(filename):
    phrases = []
    labels = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            s1, s2, sim = l.split(',')
            phrases.append((s1.strip(' '), s2.strip(' ')))
            labels.append(float(sim.strip('\n ')))
    return phrases, labels





def run(input, attention_mask, seg, target, mode='train'):
    if mode == 'train':
        model.train()
        model.zero_grad()
    else:
        model.eval()

    h_0 = torch.rand(2, input.shape[0], args.lstm_hidden_size).to(device)
    c_0 = torch.rand(2, input.shape[0], args.lstm_hidden_size).to(device)

    output = model(input, attention_mask, seg, h_0, c_0)
    loss = criterion(output, target.unsqueeze(-1))

    if mode == 'train':
        loss.backward()
        optimizer.step()

    return loss.item()






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", type=str, required=True, help="Path to training/dev/test data")
    parser.add_argument("--output", type=str, required=True, help="Path to output")
    
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of minibatches")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.0004, help="Weight decay")
    parser.add_argument("--split_train", type=float, default=0.6, help="Split percentage of train portion")
    parser.add_argument("--split_dev", type=float, default=0.2, help="Split percentage of dev portion")
    parser.add_argument("--bert_hidden_size", type=int, default=768, help="Hidden size of BERT embeddings")
    parser.add_argument("--lstm_hidden_size", type=int, default=128, help="Hidden size of LSTM embeddings")
    parser.add_argument("--classifier_hidden_size", type=int, default=256, help="Hidden size of the classifier")
    parser.add_argument("--max_padding_length", type=int, default=30, help="Maximum length of padding of BERT tokenizer")
    parser.add_argument("--pooling", type=str, default="mean", help="Pooling method for aggregation")

    args = parser.parse_args()

    print(vars(args))

    # Define the GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(device)

    print_every = 1

    phrases, labels = read_training_data(args.data)
    assert len(phrases) == len(labels)
    num_training_example = len(phrases)
    print('Training dataset read')


    model = Conceptual_Similarity(bert_hidden_size=args.bert_hidden_size, lstm_hidden_size=args.lstm_hidden_size, classifier_hidden_size=args.classifier_hidden_size, dropout=args.dropout, pooling=args.pooling)
    criterion = nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    tokenizer = model.tokenizer

    x_train = []
    attention_mask_train = []
    seg_train = []


    for i in range(len(phrases)):
        tmp = tokenizer.encode_plus(phrases[i][0], phrases[i][1], add_special_tokens=True, padding='max_length', max_length=args.max_padding_length)
        x_train.append(tmp['input_ids'])
        attention_mask_train.append(tmp['attention_mask'])
        seg_train.append(tmp['token_type_ids'])



    x_train = torch.tensor(x_train)
    attention_mask_train = torch.tensor(attention_mask_train)
    seg_train = torch.tensor(seg_train)
    y_train = torch.tensor(labels)

    # Splitting the data
    boundary_train = int(num_training_example * args.split_train)
    boundary_dev = boundary_train + int(num_training_example * args.split_dev)

    x_dev = x_train[boundary_train: boundary_dev]
    attention_mask_dev = attention_mask_train[boundary_train: boundary_dev]
    seg_dev = seg_train[boundary_train: boundary_dev]
    y_dev = y_train[boundary_train: boundary_dev]


    x_test = x_train[boundary_dev:]
    attention_mask_test = attention_mask_train[boundary_dev:]
    seg_test = seg_train[boundary_dev:]
    y_test = y_train[boundary_dev:]

    x_train = x_train[:boundary_train]
    attention_mask_train = attention_mask_train[:boundary_train]
    seg_train = seg_train[:boundary_train]
    y_train = y_train[:boundary_train]
    print("Dataset processing done")



    train_loss = 0
    dev_loss = 0
    test_loss = 0

    model.train()
    model.to(device)


    num_training_examples = x_train.shape[0]
    num_batches = math.ceil(num_training_examples / args.batch_size)

    num_dev_examples = x_dev.shape[0]
    num_batches_dev = math.ceil(num_dev_examples / args.batch_size)

    num_test_examples = x_test.shape[0]
    num_batches_test = math.ceil(num_test_examples / args.batch_size)


    hyperparameters = {
            'learning_rate': args.lr,
            'n_epochs': args.epochs,
            'minibatch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'dropout': args.dropout,
            'split_train': args.split_train,
            'split_dev': args.split_dev,
            'bert_hidden_size': args.bert_hidden_size,
            'lstm_hidden_size': args.lstm_hidden_size,
            'classifier_hidden_size': args.classifier_hidden_size,
            'loss': 'BCELoss',
            'optimizer': 'Adam',
            'random_seed': 0,
            'torch_seed': 0,
            'num_examples': len(phrases),
            'max_padding_length': args.max_padding_length,
            'pooling': args.pooling
        }

    # To save the best performoing epoch on the dev set
    best_epoch_train = 0
    best_epoch_dev = 0
    best_epoch_test = 0
    min_loss_train = float('inf')
    min_loss_dev = float('inf')
    min_loss_test = float('inf')

    for epoch in range(args.epochs):
        for i in tqdm.tqdm(range(num_batches)):
            boundary = i * args.batch_size
            sentence = x_train[boundary: boundary + args.batch_size].to(device)
            attention_mask = attention_mask_train[boundary: boundary + args.batch_size].to(device)
            seg = seg_train[boundary: boundary + args.batch_size].to(device)
            target = y_train[boundary: boundary + args.batch_size].to(device)

            train_loss += run(sentence, attention_mask, seg, target, mode='train')


        for i in range(num_batches_dev):
            boundary = i * args.batch_size
            sentence = x_dev[boundary: boundary + args.batch_size].to(device)
            attention_mask = attention_mask_dev[boundary: boundary + args.batch_size].to(device)
            seg = seg_dev[boundary: boundary + args.batch_size].to(device)
            target = y_dev[boundary: boundary + args.batch_size].to(device)

            dev_loss += run(sentence, attention_mask, seg, target, mode='dev')


        for i in range(num_batches_test):
            boundary = i * args.batch_size
            sentence = x_test[boundary: boundary + args.batch_size].to(device)
            attention_mask = attention_mask_test[boundary: boundary + args.batch_size].to(device)
            seg = seg_test[boundary: boundary + args.batch_size].to(device)
            target = y_test[boundary: boundary + args.batch_size].to(device)

            test_loss += run(sentence, attention_mask, seg, target, mode='test')


        if dev_loss < min_loss_dev:
            min_loss_dev = dev_loss
            best_epoch_dev = epoch
            hyperparameters['n_epochs'] = epoch + 1

            torch.save({
                'model': model.state_dict(),
                'hyperparameters': hyperparameters,
                'epoch': epoch
            } , args.output + "_dev.pt")


        
        if train_loss < min_loss_train:
            min_loss_train = train_loss
            best_epoch_train = epoch
            hyperparameters['n_epochs'] = epoch + 1

            torch.save({
                'model': model.state_dict(),
                'hyperparameters': hyperparameters,
                'epoch': epoch
            } , args.output + "_train.pt")


        
            
        if (epoch + 1) % print_every == 0:
            train_loss /= (num_training_examples * print_every)
            dev_loss /= (num_dev_examples * print_every)
            test_loss /= (num_test_examples * print_every)
            print("Training {:.2f}% --> Training Loss = {:.4f}".format(round(((epoch + 1) / args.epochs) * 100, 2), train_loss))
            print("Training {:.2f}% --> Dev Loss = {:.4f}".format(round(((epoch + 1) / args.epochs) * 100, 2), dev_loss))
            print("Training {:.2f}% --> Test Loss = {:.4f}".format(round(((epoch + 1) / args.epochs) * 100, 2), test_loss))
            print("This epoch: {}  |  Train: {}  |  Dev: {}  |  Test: {}".format(epoch, best_epoch_train, best_epoch_dev, best_epoch_test))
            print()
            train_loss = 0
            dev_loss = 0
            test_loss = 0


    print("Training Complete")