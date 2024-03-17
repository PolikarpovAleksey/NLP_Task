import torch
import torch.nn.functional as F
import os
from colorama import Fore, Style, init
import sys
init()

def build_dataset(words, stoi, block_size=3):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi.get(ch, 0)  # Если символа нет в словаре, используйте индекс 0
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

def test(model_file, test_file):
    # Загрузка параметров модели из файла
    parameter = torch.load(model_file)
    C = parameter['C']
    W1 = parameter['W1']
    b1 = parameter['b1']
    W2 = parameter['W2']
    b2 = parameter['b2']

    # Загрузка словаря символов
    stoi = torch.load('stoi.pth')
    itos = {i: s for s, i in stoi.items()}

    # Построение пути к файлу с данными для тестирования
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(current_dir, test_file)

    # Загрузка данных для тестирования из файла
    args = sys.argv
    words = open(args[2], "r", encoding="utf-8").read().splitlines()

    # Построение датасета для тестирования
    X_test, Y_test = build_dataset(words, stoi)

    # Вычисление потерь на тестовых данных
    emb = C[X_test]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_test)

    print(Style.BRIGHT + f'test loss: {loss.item():.2f}')
    print('example:')

    g = torch.Generator().manual_seed(2147483645 + 10)
    for _ in range(5):
        out = []
        context = [0] * 3
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break

        print(Fore.CYAN + ''.join(itos[i] for i in out))

# Пример использования функции
test('model.pth', 'test.txt')
