#!/usr/bin/env python3
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import random
import time
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import Models

def findFiles(path): return glob.glob(path)


import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"

# Turn a Unicode string to plain ASCII,
# thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines


# -----------------------
# Turn names into tensors
# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters> tensor
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l, lower =0, upper = 1):
    return l[random.randint(lower, len(l) - upper)]


def randomTrainingExampleTR():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category],upper=50)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def randomTrainingExampleVA():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category],lower=50)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


n_letters = len(all_letters)
n_hidden = 128
n_categories = len(all_categories)

steps = 1000
learning_rate = 0.001


model = Models.SumRNN(n_letters, n_hidden, num_classes = n_categories).to(device) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


floating = 0

for i in range(1,steps+1):
    category, line, category_tensor, line_tensor = randomTrainingExampleTR()
    line_tensor = line_tensor.permute(1, 0, 2).to(device) 
    category_tensor = category_tensor.to(device) 

    outputs = model(line_tensor)

    loss = criterion(outputs, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    floating += loss.item()
    if (i+1) % 100 == 0:
        print (f'Loss: {floating/100:.4f}, {100*i/steps:.2f}% Complete')
        floating = 0



with torch.no_grad():
    correct = 0
    samples = 250
    for i in range(1,samples):
        category, line, category_tensor, line_tensor = randomTrainingExampleVA()
        line_tensor = line_tensor.permute(1, 0, 2).to(device) 
        category_tensor = category_tensor.to(device) 

        outputs = model(line_tensor)
        samples += 1

        if torch.max(outputs.data, 1)[1] == category_tensor:
            correct += 1

    print(f'{100*correct/samples:.2f}% Correct')
    
