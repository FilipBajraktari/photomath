import os
import cv2
import numpy as np
import preprocessing as prep

org_dir = '/home/filip/Desktop/informatika/Petnica_project_2020-21/extracted_images'
where_dir = '/home/filip/Desktop/informatika/Petnica_project_2020-21/inverted_images'

for symbol in os.listdir(org_dir):
    
    where_symbol_dir = os.path.join(where_dir, symbol)
    if not os.path.exists(where_symbol_dir):
        os.mkdir(where_symbol_dir, mode=0o755)

    org_symbol_dir = os.path.join(org_dir, symbol)
    for filename in os.listdir(org_symbol_dir):

        org_filename_dir = os.path.join(org_symbol_dir, filename)
        img = cv2.imread(org_filename_dir)
        img = prep.processing(img)

        where_filename_dir = os.path.join(where_symbol_dir, filename)
        cv2.imwrite(where_filename_dir, img)

    print(symbol)
   