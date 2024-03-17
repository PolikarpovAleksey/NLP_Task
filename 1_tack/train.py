import torch
from tqdm import tqdm
import torch.nn.functional as F
from colorama import Fore, Style, init
import sys
init()

def build_dataset(words, block_size=3):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

# Чтение данных из файла train.txt
args = sys.argv
words = open(args[1], "r", encoding="utf-8").read().splitlines()

n = int(0.9 * len(words))

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0

torch.save(stoi, 'stoi.pth')

Xtr, Ytr = build_dataset(words[:n])
Xdev, Ydev = build_dataset(words[n:])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((34, 10), generator=g)
W1 = torch.randn((30, 50), generator=g)
b1 = torch.randn(50, generator=g)
W2 = torch.randn((50, 34), generator=g)
b2 = torch.randn(34, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre

iterations = 20_000
lossi = []
stepi = []

for i in tqdm(range(iterations)):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (34,))

    # forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    stepi.append(i)
    lossi.append(loss.log10().item())

torch.save({
    'C': C,
    'W1': W1,
    'b1': b1,
    'W2': W2,
    'b2': b2
}, 'model.pth')
print(Fore.GREEN + Style.BRIGHT + 'создан файл model.pth')
