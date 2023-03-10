import sys
sys.path.append('./PyTorch')

from PIL import Image, ImageDraw
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from PyTorch import symbols
from PyTorch import neuralNetwork as nn
from modules import preprocessing as prep
from modules import inkmlTruth

from scipy.ndimage.measurements import label


def printArray(array):
    h, w = array.shape
    for i in range(h):
        for j in range(w):
            print(array[i][j], end=' ')
        print(end='\n')

#INITIALIZE
net = nn.Net()
dir_of_model = '/home/filip/Desktop/informatika/Petnica_project_2020-21/PyTorch/net.pth'
net.load_state_dict(torch.load(dir_of_model))
print('Net loaded!\n')


def Prediction(img):
    img = img.astype(np.uint8)
    img = np.reshape(img, (45, 45, 1))
    img[img > 0] = 255

    transform = ToTensor()
    img = transform(img)
    img = torch.reshape(img, (1, 1, 45, 45))

    outputs = net(img)
    _, predicted = torch.max(outputs.data, 1)
    return symbols.number2symbol(predicted.item())


img = Image.open('./dataset_jpg/training/49.jpg')
img_array = prep.processing(np.asarray(img))
img_array = prep.extraDenoising(img_array)

h, w = img_array.shape

img = Image.fromarray(img_array)
img_draw = ImageDraw.Draw(img)

niz_simbola = []
dugacki_simboli = ['pm', 'in', 'mu', 'pi', 'to',
                   'sin', 'cos', 'tan', 'lim', 'neq', 'leq', 'geq', 'phi', 'int', 'sum',
                   'beta', 'frac',
                   'theta', 'alpha', 'gamma', 'delta', 'infty', 'sigma', 'times'
                   'lambda', 'forall', 'exists']

#(simbol, [leva_granica, desna_granica, gornja_granica, donja_granica])
def preklopljeniSimboli(gornji_levi, donji_desni, srt='levo'):
    j_start, i_start = gornji_levi
    j_end, i_end = donji_desni

    kernel = np.ones((3, 3))
    a = img_array[i_start:i_end+1, j_start:j_end+1]
    labeled_array, num_features = label(a, kernel)

    okviri = [[w, 0, h, 0] for i in range(num_features)]
    mape = [np.zeros((h, w)) for i in range(num_features)]
    for i in range(i_start, i_end+1, 1):
        for j in range(j_start, j_end+1, 1):
            i_r = i - i_start
            j_r = j - j_start
            broj = labeled_array[i_r][j_r]
            
            if broj > 0:
                okviri[broj-1][0] = min(okviri[broj-1][0], j)
                okviri[broj-1][1] = max(okviri[broj-1][1], j)
                okviri[broj-1][2] = min(okviri[broj-1][2], i)
                okviri[broj-1][3] = max(okviri[broj-1][3], i)

                mape[broj-1][i][j] = 255

    ret = []
    for nbr, okvir in enumerate(okviri, 0):

        simbol_array = prep.makeSquare(mape[nbr][okvir[2]:okvir[3]+1, okvir[0]:okvir[1]+1])
        simbol = Prediction(simbol_array)

        #niz_simbola.append(simbol_array)
        img_draw.rectangle([okvir[0], okvir[2], okvir[1], okvir[3]], fill=None, outline='green')
        ret.append((simbol, okvir))

    if srt == 'levo':
        ret.sort(key=lambda x:x[1][0])
    elif srt == 'gore':
        ret.sort(key=lambda x:x[1][2])

    return ret

def x_segmentation(gornji_levi, donji_desni, start=True):
    j_start, i_start = gornji_levi
    j_end, i_end = donji_desni

    if prep.onePeak(img_array[i_start:i_end+1, j_start:j_end+1]):
        '''
        levo, desno = prep.x_shrink(
            img_array[i_start:i_end+1, j_start:j_end+1])
        # img_draw.rectangle([j_start+levo-1, i_start-1, j_end -
        #                    desno+1, i_end+1], fill=None, outline='green')

        symbol_array = prep.makeSquare(
            img_array[i_start:i_end+1, j_start+levo:j_end-desno+1])
        niz_simbola.append([j_start+levo, j_end-desno, i_start, i_end])
        symbol = Prediction(symbol_array)

        #(simbol, [leva granica, desna granica, gornja granica, donja granica])
        return [(symbol, [j_start+levo, j_end-desno+1, i_start, i_end+1])]
        '''
        if start == False:
            return preklopljeniSimboli(gornji_levi, donji_desni, 'levo')
        

    x_cord = prep.x_coordinate(img_array[i_start:i_end, j_start:j_end])
    truth = ""
    poz = 0 #0 je za baseline, 1 je za stepen, -1 je za index
    baseline = 0
    lower_bound, upper_bound = 0, h
    glavni_okvir = [10000, 0, 10000, 0]

    koren = 0
    granica_koren = []
    for i, (start, end) in enumerate(x_cord, 0):
        if start == False or len(x_cord) > 1:
            y_simboli = y_segmentation((j_start+start, i_start), (j_start+end, i_end))
        else:
            y_simboli = preklopljeniSimboli(gornji_levi, donji_desni, srt='levo')

        for j, (simbol, granice) in enumerate(y_simboli, 0):

            glavni_okvir[0] = min(glavni_okvir[0], granice[0])
            glavni_okvir[1] = max(glavni_okvir[1], granice[1])
            glavni_okvir[2] = min(glavni_okvir[2], granice[2])
            glavni_okvir[3] = max(glavni_okvir[3], granice[3])

            #PROSTORNA PROVERA
            if not (i == 0 and j == 0) and truth[-5:]!='sqrt{' and simbol!='-':
                if granice[3] < baseline:
                    if poz >= 0:
                        truth += '^{'
                        poz += 1
                    elif poz < 0:
                        truth += '}'
                        poz += 1
                elif baseline < granice[2]:
                    if poz <= 0:
                        truth += '_{'
                        poz -= 1
                    elif poz > 0:
                        truth += '}'
                        poz -= 1


            if simbol in dugacki_simboli:
                truth += '\\' + simbol
            elif simbol == 'lt':
                truth += '<'
            elif simbol == 'gt':
                truth += '>'
            elif simbol == 'forward_slash':
                truth += '/'
            elif simbol == 'ascii_124':
                truth += '|'
            elif simbol == ',':
                truth += ')'
            else:
                #PROVERA ZA KOREN
                if simbol == 'sqrt':
                    truth += '\\sqrt' + '{'
                    koren += 1
                    granica_koren.append(granice[1])
                else:
                    if koren > 0:
                        if granica_koren[-1] < granice[0]:
                            truth += '}'
                            koren -= 1
                            granica_koren.pop()
                    truth += simbol

            #TRACK OF BASELINE
            _,  ycm = prep.centreOfMass(img_array[granice[2]:granice[3]+1, granice[0]:granice[1]+1])
            baseline = granice[2] + ycm
            lower_bound = granice[2]-1
            upper_bound = granice[3]+1

    while poz > 0:
        truth += '}'
        poz -= 1
    while koren > 0:
        truth += '}'
        koren -= 1
        granica_koren.pop()

            
    i = 0
    while i<len(truth):
        if truth[i] == '\\':
            i += 2
        #SIN
        elif i+2<len(truth) and (truth[i]=='s'or truth[i]=='5')and(truth[i+1]=='i'or truth[i+1]=='|'or truth[i+1]=='1')and truth[i+2]=='n':
            truth = truth[:i]+'\\sin'+truth[i+3:]
            i += 3
        #SIN
        elif i+1<len(truth) and truth[i]=='s' and truth[i+1]=='m':
            truth = truth[:i]+'\\sin'+truth[i+2:]
            i += 3
        #COS
        elif i+2<len(truth) and truth[i]=='c' and (truth[i+1]=='o' or truth[i+1]=='0') and (truth[i+2]=='s' or truth[i+2]=='5'):
            truth = truth[:i]+'\\cos'+truth[i+3:]
            i += 3
        #TAN
        elif i+2<len(truth) and truth[i]=='t' and (truth[i+1]=='a' ) and (truth[i+2]=='n'):
            truth = truth[:i]+'\\tan'+truth[i+3:]
            i += 3
        #LIM
        elif i+2<len(truth) and truth[i]=='l' and (truth[i+1]=='i'or truth[i+1]=='|'or truth[i+1]=='1') and (truth[i+2]=='m'):
            truth = truth[:i]+'\\tan'+truth[i+3:]
            i += 3
        #RIGHTARROW / TO
        elif i+1<len(truth) and truth[i]=='-' and truth[i+1]=='>':
            truth = truth[:i]+'\\rightarrow'+truth[i+2:]
            i += 10
        
        i += 1

    return [(truth, glavni_okvir)]


def y_segmentation(gornji_levi, donji_desni):
    j_start, i_start = gornji_levi
    j_end, i_end = donji_desni

    if prep.onePeak(img_array[i_start:i_end+1, j_start:j_end+1]):
        '''
        gore, dole = prep.y_shrink(img_array[i_start:i_end+1, j_start:j_end+1])
        # img_draw.rectangle([j_start-1, i_start+gore-1, j_end+1,
        #                    i_end-dole+1], fill=None, outline='green')

        symbol_array = prep.makeSquare(
            img_array[i_start+gore:i_end-dole+1, j_start:j_end+1])
        niz_simbola.append([j_start, j_end, i_start+gore, i_end-dole])
        symbol = Prediction(symbol_array)
        
        #(simbol, [leva granica, desna granica, gornja granica, donja granica])
        return [(symbol, [j_start, j_end+1, i_start+gore, i_end-dole+1])]
        '''
        return preklopljeniSimboli(gornji_levi, donji_desni, 'levo')
        
        

    y_cord = prep.y_coordinate(img_array[i_start:i_end, j_start:j_end])
    y_simboli = []
    glavni_okvir = [10000, 0, 10000, 0]
    for i, (start, end) in enumerate(y_cord, 0):
        simboli = x_segmentation((j_start, i_start+start), (j_end, i_start+end), start=False)

        for i, (expresion, granice) in enumerate(simboli, 0):
            glavni_okvir[0] = min(glavni_okvir[0], granice[0])
            glavni_okvir[1] = max(glavni_okvir[1], granice[1])
            glavni_okvir[2] = min(glavni_okvir[2], granice[2])
            glavni_okvir[3] = max(glavni_okvir[3], granice[3])

            y_simboli.append((expresion, granice))

    n = len(y_simboli)
    truth = ""
    ret = []
    if n == 2:
        if y_simboli[0][0] == '\lim':
            truth = '\lim_' + '{' + y_simboli[1][0] + '}'
            ret.append((truth, glavni_okvir))
        elif y_simboli[0][0] == '+' and y_simboli[1][0] == '-':
            truth = '\pm'
            ret.append((truth, glavni_okvir))
        elif y_simboli[0][0] == '-' and y_simboli[1][0] == '-':
            truth = '='
            ret.append((truth, glavni_okvir))
        elif y_simboli[0][0] == '1' and y_simboli[1][0] == '-':
            truth = '1'
            ret.append((truth, glavni_okvir))
        elif (y_simboli[0][0]=='<' and y_simboli[1][0]=='-') or (y_simboli[0][0]=='-' and y_simboli[1][0]=='<'):
            truth = '\leq'
            ret.append((truth, glavni_okvir))
        elif (y_simboli[0][0]=='>' and y_simboli[1][0]=='-') or (y_simboli[0][0]=='-' and y_simboli[1][0]=='>'):
            truth = '\qeq'
            ret.append((truth, glavni_okvir))
    elif n == 3:
        if y_simboli[1][0] == '-':
            truth = '\\frac' + '{' + y_simboli[0][0] + '}{' + y_simboli[2][0] + '}'
            ret.append((truth, glavni_okvir))
        elif y_simboli[1][0] == '\int':
            truth = '\int_' + '{' + y_simboli[2][0] + '}{' + y_simboli[0][0] + '}'
            ret.append((truth, glavni_okvir))
        elif y_simboli[1][0] == '\sum':
            truth = '\sum_' + '{' + y_simboli[2][0] + '}{' + y_simboli[0][0] + '}'
            ret.append((truth, glavni_okvir))

    if len(ret) == 0: 
        y_simboli.sort(key=lambda x:x[1][0])
        return y_simboli
    else: return ret

#izlaz je (izraz, glavni okvir)
if prep.y_peak(img_array)>1 and prep.x_peak(img_array)==1:
    izlaz = y_segmentation((0, 0), (w-2, h-2))
else:
    izlaz = x_segmentation((0, 0), (w-2, h-2), start=True)
print(end='\n')
print(izlaz[0][0])
prep.showLatex(izlaz[0][0])
img.show()

# out = preklopljeniSimboli((0, 0), (w-1, h-1), srt='levo')
# for i in range(7):
#     print(out[i][0])

# correct = 0
# total = 0
# greske = 0
# duzina = 0
# #plotovanje grafika
# x = np.arange(0, 50)
# y = np.zeros((50,))
# for filename in os.listdir('./dataset_jpg/validation'):
#     img = Image.open(os.path.join('./dataset_jpg/validation', filename))
#     img_array = prep.processing(np.asarray(img))
#     img_array = prep.extraDenoising(img_array)

#     h, w = img_array.shape

#     # img = Image.fromarray(img_array)
#     # img_draw = ImageDraw.Draw(img)

#     if prep.y_peak(img_array)>1 and prep.x_peak(img_array)==1:
#         izlaz = y_segmentation((0, 0), (w-2, h-2))
#     else:
#         izlaz = x_segmentation((0, 0), (w-2, h-2), start=True)
#     labela = inkmlTruth.Truth(os.path.join('./dataset_inkml/validation', f'{filename[:-4]}.inkml'))

#     if prep.isEqual(izlaz[0][0], labela):
#         correct += 1
#     # else:
#     #     print(izlaz[0][0])
#     #     print(labela, end='\n\n')
#     total += 1

#     trenutnaGreska = prep.editDistance(izlaz[0][0], labela)
#     greske += trenutnaGreska
#     duzina += len(labela)

#     #plotovanje grafika
#     if trenutnaGreska<50:
#         y[trenutnaGreska]+=1

#     # img.show()
#     # os.wait()

# print('Correct: {0}'.format(correct))
# print('Total: {0}'.format(total))
# print('Greske: {0}'.format(greske/duzina))
# img.show()
# #plotovanje grafika
# plt.xlabel('Broj gresaka')
# plt.ylabel('Broj primera')
# plt.bar(x,y, align='center')
# plt.show()


# correct = 0
# total = 0
# greske = 0
# duzina = 0
# dozvoljeni = []
# #plotovanje grafika
# x = np.arange(0, 50)
# y = np.zeros((50,))
# with open('./dataset_inkml/validation.txt') as file:
#     dozvoljeni = file.readlines()

# for filename in os.listdir('./dataset_jpg/validation'):
#     if filename[:-3]+'inkml\n' in dozvoljeni:
#         img = Image.open(os.path.join('./dataset_jpg/validation', filename))
#         img_array = prep.processing(np.asarray(img))
#         img_array = prep.extraDenoising(img_array)

#         h, w = img_array.shape

#         # img = Image.fromarray(img_array)
#         # img_draw = ImageDraw.Draw(img)

#         if prep.y_peak(img_array)>1 and prep.x_peak(img_array)==1:
#             izlaz = y_segmentation((0, 0), (w-2, h-2))
#         else:
#             izlaz = x_segmentation((0, 0), (w-2, h-2), start=True)
#             labela = inkmlTruth.Truth(os.path.join('./dataset_inkml/validation', f'{filename[:-4]}.inkml'))

#         if prep.isEqual(izlaz[0][0], labela):
#             correct += 1
#         # else:
#         #     print(izlaz[0][0])
#         #     print(labela, end='\n\n')
#         total += 1

#         trenutnaGreska = prep.editDistance(izlaz[0][0], labela)
#         greske += trenutnaGreska
#         duzina += len(labela)

#         #plotovanje grafika
#         if trenutnaGreska<50:
#             y[trenutnaGreska]+=1

#         # img.show()
#         # os.wait()

# print('Correct: {0}'.format(correct))
# print('Total: {0}'.format(total))
# print('Greske: {0}'.format(greske/duzina))
# img.show()
# #plotovanje grafika
# plt.xlabel('Broj gresaka')
# plt.ylabel('Broj primera')
# plt.bar(x,y, align='center')
# plt.show()