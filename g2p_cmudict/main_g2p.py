import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.data as data

from g2p_cmudict.cmu_dict import CMUDict
from g2p_cmudict.g2p_model import G2P
from g2p_cmudict.utils import adjust_learning_rate, phoneme_error_rate


def train(config, train_iter, model, criterion, optimizer, epoch):
    global iteration, n_total, train_loss, n_bad_loss
    global init, best_val_loss, stop

    print("=> EPOCH {}".format(epoch))
    train_iter.init_epoch()
    for batch in train_iter:
        iteration += 1
        model.train()

        output, _, __ = model(batch.grapheme, batch.phoneme[:-1].detach())
        target = batch.phoneme[1:]
        loss = criterion(output.view(output.size(0) * output.size(1), -1),
                         target.view(target.size(0) * target.size(1)))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), config.clip, 'inf')
        optimizer.step()

        n_total += batch.batch_size
        train_loss += loss.data * batch.batch_size

        if iteration % config.log_every == 0:
            train_loss /= n_total
            val_loss = validate(val_iter, model, criterion)
            print("   % Time: {:5.0f} | Iteration: {:5} | Batch: {:4}/{}"
                  " | Train loss: {:.4f} | Val loss: {:.4f}"
                  .format(time.time() - init, iteration, train_iter.iterations,
                          len(train_iter), train_loss, val_loss))

            # test for val_loss improvement
            n_total = train_loss = 0
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                n_bad_loss = 0
                torch.save(model.state_dict(), config.best_model)
            else:
                n_bad_loss += 1
            if n_bad_loss == config.n_bad_loss:
                best_val_loss = val_loss
                n_bad_loss = 0
                adjust_learning_rate(optimizer, config.lr_decay)
                new_lr = optimizer.param_groups[0]['lr']
                print("=> Adjust learning rate to: {}".format(new_lr))
                if new_lr < config.lr_min:
                    stop = True
                    break


def validate(val_iter, model, criterion):
    model.eval()
    val_loss = 0
    val_iter.init_epoch()
    for batch in val_iter:
        output, _, __ = model(batch.grapheme, batch.phoneme[:-1])
        target = batch.phoneme[1:]
        loss = criterion(output.squeeze(1), target.squeeze(1))
        val_loss += loss.data * batch.batch_size
    return val_loss / len(val_iter.dataset)


def test(test_iter, model, criterion):
    model.eval()
    test_iter.init_epoch()
    test_per = test_wer = 0
    for batch in test_iter:
        output = model(batch.grapheme).data.tolist()
        target = batch.phoneme[1:].squeeze(1).data.tolist()
        # calculate per, wer here
        per = phoneme_error_rate(output, target)
        wer = int(output != target)
        test_per += per  # batch_size = 1
        test_wer += wer

    test_per = test_per / len(test_iter.dataset) * 100
    test_wer = test_wer / len(test_iter.dataset) * 100
    print("Phoneme error rate (PER): {:.2f}\nWord error rate (WER): {:.2f}"
          .format(test_per, test_wer))


def show(batch, model):
    assert batch.batch_size == 1
    g_field = batch.dataset.fields['grapheme']
    p_field = batch.dataset.fields['phoneme']
    prediction = model(batch.grapheme).data.tolist()[:-1]
    grapheme = batch.grapheme.squeeze(1).data.tolist()[1:][::-1]
    phoneme = batch.phoneme.squeeze(1).data.tolist()[1:-1]
    print("> {}\n= {}\n< {}\n".format(
        ''.join([g_field.vocab.itos[g] for g in grapheme]),
        ' '.join([p_field.vocab.itos[p] for p in phoneme]),
        ' '.join([p_field.vocab.itos[p] for p in prediction])))


parser = {
    'data_path': '../data/cmudict/',
    'epochs': 15,
    'batch_size': 100,
    'max_len': 20,  # max length of grapheme/phoneme sequences
    'beam_size': 3,  # size of beam for beam-search
    'd_embed': 500,  # embedding dimension
    'd_hidden': 500,  # hidden dimension
    'attention': True,  # use attention or not
    'log_every': 100,  # number of iterations to log and validate training
    'lr': 0.007,  # initial learning rate
    'lr_decay': 0.5,  # decay lr when not observing improvement in val_loss
    'lr_min': 1e-5,  # stop when lr is too low
    'n_bad_loss': 5,  # number of bad val_loss before decaying
    'clip': 2.3,  # clip gradient, to avoid exploding gradient
    'cuda': False,  # using gpu or not
    'seed': 5,  # initial seed
    'intermediate_path': '../intermediate/g2p/',  # path to save models
}
args = argparse.Namespace(**parser)

args.cuda = args.cuda and torch.cuda.is_available()

if not os.path.isdir(args.intermediate_path):
    os.makedirs(args.intermediate_path)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Tokenize cmudict
g_field = data.Field(init_token='<s>',
                     tokenize=(lambda x: list(x.split('(')[0])[::-1]))
p_field = data.Field(init_token='<os>', eos_token='</os>',
                     tokenize=(lambda x: x.split('#')[0].split()))

filepath = os.path.join(args.data_path, 'cmudict.dict')
train_data, val_data, test_data = CMUDict.splits(filepath, g_field, p_field,
                                                 args.seed)

g_field.build_vocab(train_data, val_data, test_data)
p_field.build_vocab(train_data, val_data, test_data)

device = None if args.cuda else -1  # None is current gpu
train_iter = data.BucketIterator(train_data, batch_size=args.batch_size,
                                 repeat=False, device=device)
val_iter = data.Iterator(val_data, batch_size=1,
                         train=False, sort=False, device=device)
test_iter = data.Iterator(test_data, batch_size=1,
                          train=False, shuffle=True, device=device)

config = args
config.g_size = len(g_field.vocab)
config.p_size = len(p_field.vocab)
config.best_model = os.path.join(config.intermediate_path,
                                 "best_model_adagrad_attn.pth")

model = G2P(config)
criterion = nn.NLLLoss()
if config.cuda:
    model.cuda()
    criterion.cuda()
optimizer = optim.Adagrad(model.parameters(), lr=config.lr)  # use Adagrad

# if 1 == 1:  # change to True to train
#     iteration = n_total = train_loss = n_bad_loss = 0
#     stop = False
#     best_val_loss = 10
#     init = time.time()
#     for epoch in range(1, config.epochs+1):
#         train(config, train_iter, model, criterion, optimizer, epoch)
#         if stop:
#             break


model.load_state_dict(torch.load(config.best_model))
test(test_iter, model, criterion)
test_iter.init_epoch()
for i, batch in enumerate(test_iter):
    show(batch, model)
    # if i == 50:
    #     break
