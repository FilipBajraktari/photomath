import sys
sys.path.append('/home/filip/Desktop/informatika/Petnica_project_2020-21')

import os
import pickle
import numpy as np
from PyTorch import symbols
from inkmlTruth import Truth

#matrica je za jedan broj veca da bi se ubacila
#verovatnoca da izraz zapocinje odredjenim simbolom
matrica = np.zeros((83, 83))

def index(simbol):
    if simbol[0] == '\\':
        return symbols.symbol2number(simbol[1:])

    return symbols.symbol2number(simbol)

def updateTableClean(truth):
    niz_simbola = []
    n = len(truth)
    i = 0
    while i<n:
        if truth[i] == '\\':
            if i+2<n and (truth[i+1:i+3] in ['pm', 'in', 'mu', 'pi', 'to', 'gt', 'lt']):
                if truth[i+1:i+3] == 'to':
                    niz_simbola.append('rightarrow')
                    i += 2
                if truth[i+1:i+3]=='in' and i+3<n and truth[i+1:i+4]=='int':
                    niz_simbola.append(truth[i:i+4])
                    i += 3
                elif truth[i+1:i+3]=='in' and i+5<n and truth[i+1:i+6]=='infty':
                    niz_simbola.append(truth[i:i+6])
                    i += 5
                else:
                    niz_simbola.append(truth[i:i+3])
                    i += 2
            elif i+3<n and (truth[i+1:i+4] in ['sin', 'cos', 'tan', 'lim', 'log', 'neq', 'leq', 'geq', 'phi', 'int', 'sum']):
                niz_simbola.append(truth[i:i+4])
                i += 3
            elif i+4<n and (truth[i+1:i+5] in ['beta', 'frac', 'sqrt', 'left']):
                if truth[i+1:i+5] == 'left':
                    i += 5
                    continue
                niz_simbola.append(truth[i:i+5])
                i += 4
            elif i+5<n and (truth[i+1:i+6] in ['theta', 'alpha', 'gamma', 'delta', 'infty', 'sigma', 'times', 'ldots', 'right']):
                if truth[i+1:i+6] == 'right':
                    if i+10<n and truth[i+1:i+11] == 'rightarrow':
                        niz_simbola.append(truth[i:i+11])
                        i += 11
                        continue
                    else:
                        i += 6
                        continue
                niz_simbola.append(truth[i:i+6])
                i += 5
            elif i+6<n and (truth[i+1:i+7] in ['lambda', 'forall', 'exists']):
                niz_simbola.append(truth[i:i+7])
                i += 6
            elif i+10<n and truth[i+1:i+11] == 'rightarrow':
                niz_simbola.append(truth[i:i+11])
                i += 10
        else:
            if truth[i]!=' ':
                niz_simbola.append(truth[i])

        i += 1
    
    for i in range(len(niz_simbola)):
        idx = index(niz_simbola[i])

        if i == 0:
            matrica[0][idx+1] += 1
        else:
            matrica[index(niz_simbola[i-1])+1][index(niz_simbola[i])+1] += 1

        if niz_simbola[i] == '\\sin':
            matrica[0][index('s')+1] += 1
            matrica[index('s')+1][index('i')+1] += 1
            matrica[index('i')+1][index('n')+1] += 1
        elif niz_simbola[i] == '\\cos':
            matrica[0][index('c')+1] += 1
            matrica[index('c')+1][index('o')+1] += 1
            matrica[index('o')+1][index('s')+1] += 1
        elif niz_simbola[i] == '\\tan':
            matrica[0][index('t')+1] += 1
            matrica[index('t')+1][index('a')+1] += 1
            matrica[index('a')+1][index('n')+1] += 1
        elif niz_simbola[i] == '\\lim':
            matrica[0][index('l')+1] += 1
            matrica[index('l')+1][index('i')+1] += 1
            matrica[index('i')+1][index('m')+1] += 1

        print(niz_simbola[i], end=' ')
    print(end='\n')

def updateTable(truth):
    if ('{' not in truth) and ('^' not in truth) and ('_' not in truth):
        updateTableClean(truth)
        return

    n = len(truth)
    i = 0
    start, nbr = -1, 0
    #niz bez simbola koji se nalaze u viticastim zagradama
    new_truth = ''
    while i<n:
        if truth[i] == '{':
            if start == -1:
                start = i
            nbr += 1
        elif truth[i] == '}':
            nbr -= 1
            if nbr == 0:
                updateTable(truth[start+1:i])
                start = -1
        
        if nbr == 0 and (truth[i]!='{' and truth[i]!='}'):

            if truth[i] == '^':
                if truth[i+1] != '{':
                    updateTable(truth[i+1])
                    i += 1
            elif truth[i] == '_':
                if truth[i+1] != '{':
                    updateTable(truth[i+1])
                    i += 1
            else:
                new_truth += truth[i]

        i += 1
    updateTableClean(new_truth)

test = r'\lim_{z \rightarrow 0} \frac 1 {\log_a ( 1 + z )^{\frac 1 z}}'
#updateTable(test.lower())

dir = '/home/filip/Desktop/informatika/Petnica_project_2020-21/dataset_inkml/training'
i = 0
for filename in os.listdir(dir):
    file_dir = os.path.join(dir, filename)
    truth = Truth(file_dir)

    bad_chars = [' ', '$']
    for c in bad_chars:
        truth = truth.replace(c, '')
    truth = truth.lower()

    updateTable(truth)


#GIBSONOVO UZIMANJE UZORAKA
d = 82
for i in range(83):
    n = matrica[i].sum()
    for j in range(83):
        matrica[i][j] = (matrica[i][j] + 1)/(n + d)

pickle.dump(matrica, open('HMMtabela', 'wb'))
