from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import numpy as np
import cv2

'''____________________PREPROCESSING____________________'''
def processing(img):
    #cv2.imshow("Test Image", img)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Gray Test Image", gray_img)

    denoised_img = cv2.fastNlMeansDenoising(gray_img)
    #denoised_img = gray_img
    #cv2.imshow("Denoised Gray Test Image", denoised_img)

    threshold_img = cv2.adaptiveThreshold(denoised_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #cv2.imshow("Threshold image", threshold_img)

    return threshold_img



'''_________________VERTITCAL PROJECTION_________________'''
def verticalProjection(img_array):
    h, w = img_array.shape

    vertical = np.zeros((w, ))
    for i in range(h):
        for j in range(w):
            if img_array[i][j] == 255:
                vertical[j] = vertical[j] + 1

    return vertical

def x_coordinate(img_array):
    verProj = verticalProjection(img_array)
    _, w = img_array.shape

    start = 0
    if verProj[0] > 0:
        start = 0
    x_cord = []
    for i in range(w):
        if i>0 and verProj[i-1]==0 and verProj[i]>0:
            start = i
        elif i>0 and verProj[i-1]>0 and verProj[i]==0:
            x_cord.append((start, i-1))
            start = 0

    if verProj[-1]>0:
        x_cord.append((start, w-1))

    return x_cord

def x_shrink(img_array):
    verProj = verticalProjection(img_array)
    _, w = img_array.shape

    start = 0
    for i in range(w):
        if verProj[i]==0:
            start = start + 1
        else: break
    
    end = 0
    for i in range(w-1, 0, -1):
        if verProj[i]==0:
            end = end + 1
        else: break

    return (start, end)



'''_________________HORIZONTAL PROJECTION_________________'''
def horizontalProjection(img_array):
    h, w = img_array.shape

    horizontal = np.zeros((h, ), np.uint8)
    for i in range(h):
        for j in range(w):
            if img_array[i][j] == 255:
                horizontal[i] = horizontal[i] + 1

    return horizontal

def y_coordinate(img_array):
    horProj = horizontalProjection(img_array)
    h, _ = img_array.shape

    start = 0
    if horProj[0] > 0:
        start = 0
    y_cord = []
    for i in range(h):
        if i>0 and horProj[i-1]==0 and horProj[i]>0:
            start = i
        elif i>0 and horProj[i-1]>0 and horProj[i]==0:
            y_cord.append((start, i-1))
            start = 0

    if horProj[-1]>0:
        y_cord.append((start, h-1))

    return y_cord

def y_shrink(img_array):
    horProj = horizontalProjection(img_array)
    h, _ = img_array.shape

    start = 0
    for i in range(h):
        if horProj[i]==0:
            start = start + 1
        else: break
    
    end = 0
    for i in range(h-1, 0, -1):
        if horProj[i]==0:
            end = end + 1
        else: break

    return (start, end)


'''_________________PREPARATION FOR NETWORK_________________'''
def scaling(img_array):
    return cv2.resize(img_array, (45, 45), interpolation=cv2.INTER_AREA)

def addPadding(img_array):
    h, w = img_array.shape
    add_padding = np.zeros((h+2, w+2))
    add_padding[1:-1, 1:-1] = img_array
    return add_padding

def makeThin(img_array):
    
    h, w = img_array.shape
    binary_array = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if img_array[i][j] == 255:
                binary_array[i][j] = 1
    thin_array = skeletonize(binary_array)
    result = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if thin_array[i][j] == 1:
                result[i][j] = 255
    return result
    '''
    kernel = np.ones((5,5))
    erosion = cv2.erode(img_array, kernel, iterations=1)
    return erosion
    '''

def makeSquare(img_array):
    h, w = img_array.shape

    sirina = max(h,w)
    result = np.zeros((sirina, sirina))
    if h > w:
        poc = int(sirina/2)-int(w/2)
        kraj = poc + w
        result[:, poc:kraj] = img_array
    else:
        poc = int(sirina/2)-int(h/2)
        kraj = poc + h
        result[poc:kraj, :] = img_array

    result_thinning = makeThin(result)
    result_scaling = scaling(result_thinning)
    #result_thinning = makeThin(result_scaling)
    
    #return result
    return result_scaling
    #return result_thinning



'''_________________LATEX EVALUATION_________________'''
#proverava da li su dva latex koda jednaka pod pretpostavkom da 
#su topoloski tacno namesteni
def cleaning(izraz):
    izraz = izraz.lower()
    bad_chars = [' ', '{', '}', '$', '\left']
    for c in bad_chars:
        izraz = izraz.replace(c, '')
    i = 0
    n = len(izraz)
    while i < n:
        if izraz[i:i+6] == '\\right' and izraz[i:11] != '\\rightarrow':
            izraz = izraz.replace(izraz[i:i+6], '')
            n -= 6
            i -= 1
        i += 1
    return izraz

def isEqual(izraz, truth):
    izraz = cleaning(izraz)
    truth = cleaning(truth)
    return (izraz==truth)

def showLatex(izraz):
    #code = string2latex(izraz)
    code = izraz
    
    plt.text(0.5, 0.5, r'${0}$'.format(code), fontsize=30,
        horizontalalignment='center', verticalalignment='center')
    plt.show()



'''______________________OTHER______________________'''
def x_peak(img_array):
    array_ver = verticalProjection(img_array)
    nbr_ver = 0
    if array_ver[0]>0:
        nbr_ver = 1
    for i in range(len(array_ver)-1):
        if array_ver[i]==0 and array_ver[i+1]>0:
            nbr_ver = nbr_ver + 1
    return nbr_ver

def y_peak(img_array):
    array_hor = horizontalProjection(img_array)
    nbr_hor = 0
    if array_hor[0]>0:
        nbr_hor = 1
    for i in range(len(array_hor)-1):
        if array_hor[i]==0 and array_hor[i+1]>0:
            nbr_hor = nbr_hor + 1
    return nbr_hor

def onePeak(img_array):
    if x_peak(img_array) == 1 and y_peak(img_array) == 1:
        return True
    else:
        return False

#cilj joj je da ukloni nasumicne bele piksele
def extraDenoising(img_array):
    h, w = img_array.shape

    for i in range(1, h-1):
        for j in range(1, w-1):
            if img_array[i][j]==255 and img_array[i-1][j]==0 and img_array[i+1][j]==0 and img_array[i][j-1]==0 and img_array[i][j+1]==0:
                if img_array[i-1][j-1]==0 and img_array[i-1][j+1]==0 and img_array[i+1][j-1]==0 and img_array[i+1][j+1]==0:
                    img_array[i][j] = 0
    return img_array

def centreOfMass(img_array):
    horProj = horizontalProjection(img_array)
    verProj = verticalProjection(img_array)

    x_centre, y_centre = 0, 0
    ukupno = 0
    for i in range(len(horProj)):
        if horProj[i] > 0:
            y_centre = y_centre + horProj[i]*(i+1)
            ukupno = ukupno + horProj[i]

    for i in range(len(verProj)):
        if verProj[i] > 0:
            x_centre = x_centre + verProj[i]*(i+1)

    return (int(x_centre/ukupno)-1, int(y_centre/ukupno)-1)

#str1 je izlaz iz main-a
#str2 je ground truth labela
def editDistance(str1, str2):
    str1 = cleaning(str1)
    m = len(str1)
    str2 = cleaning(str2)
    n = len(str2)
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):

            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j],dp[i-1][j-1])
 
    return dp[m][n]