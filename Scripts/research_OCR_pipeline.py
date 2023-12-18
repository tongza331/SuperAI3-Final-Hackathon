import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections as coll

import pandas as pd
import pylab as pl
import seaborn as sb

import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import mode
import imutils
import os
import send2trash
from research_skew_correction import util_rotate
import keyboard as kb
import pytesseract as tess
import pythainlp as ptn
import easyocr



# def adjust_s_and_r_params(value):
#     s = cv2.getTrackbarPos('S', 'my_window')
#     r = cv2.getTrackbarPos('R', 'my_window') / 100
#     img_sharp=cv2.detailEnhance(img_bgr_rotated,sigma_s=float(s),sigma_r=float(r))
#     cv2.imshow('my_window',img_sharp)
#
# cv2.namedWindow('my_window')
# s = 100 ## 22
# r = 50 ## 13
# cv2.createTrackbar('S', 'my_window',s,200,adjust_s_and_r_params)
# cv2.createTrackbar('R', 'my_window',r,100,adjust_s_and_r_params)


# def ocr_extract_text(img_bgr_rotate, coor_14_words, img_name):
#     ## Tune params s and r of detailEnhance
#     # plt.ion()
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # s=100.0
#     # r=0.5
#     # while True:
#     #     if kb.is_pressed('n'):
#     #         s+=1
#     #     if kb.is_pressed('m'):
#     #         s-=1
#     #     if kb.is_pressed(','):
#     #         r+=0.01
#     #     if kb.is_pressed('.'):
#     #         r-=0.01
#     #     img_sharp = cv2.detailEnhance(img_bgr_rotate, sigma_s=float(s), sigma_r=float(r))
#     #     ax.cla()
#     #     ax.imshow(img_sharp)
#     #     fig.canvas.draw()
#     #     fig.canvas.flush_events()
#     #     print(f'S = {s} || R = {r}')
#     # img_sharp = cv2.detailEnhance(img_bgr_rotate, sigma_s=float(s), sigma_r=float(r))
#
#     ## Trackbar adjust S and R params
#     # cv2.imshow('original',img_bgr)
#     # img_sharp=cv2.detailEnhance(img_bgr_rotate,sigma_s=float(s),sigma_r=float(r/100))
#     # cv2.imshow('my_window',img_sharp)
#     # cv2.waitKey(0)
#     border = 13
#     for pos, (x, y, w, h) in enumerate(coor_14_words):
#         img_each_word = img_bgr_rotate[y - border:y + h + border, x - border:x + w + border]
#         text = tess.image_to_string(img_each_word, lang='tha')
#         ## Show cropped image
#         # print(f'{img_name} : {text}')
#         # plt.imshow(img_each_word[:,:,::-1])
#         # plt.show()
#
#         ## Export text predict to excel
#         dict_text['img'].append(img_name + '_' + str(pos + 1))
#         dict_text['pure_text'].append(text)


def split_each_word():
    # global img_bgr_rotated,img_bgr
    true_label_list = pd.read_excel('best_cleaned_by_hand.xlsx', usecols='B')['Text'].to_list()
    tess.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    dict_text = {'img': [], 'true_label': [], 'pure_text': [], 'text_cleaned': [], 'text_correct': []}
    path_folder_img = r'C:\Users\Admin\Desktop\pythonProject\OTHER_RESEARCH\OCR\images\images'
    path_all_img = glob.glob(path_folder_img + '/*')
    if not os.path.exists('visualize'): os.mkdir('visualize')
    count_word = 0
    for pos, path_each_img in enumerate(path_all_img):
        img_name = os.path.basename(path_each_img)
        # print(f'Round : {pos + 1}')
        ##Loop by condition
        # if img_name != '00352.jpg':
        #     continue
        if pos > 10:
            break

        img_bgr = cv2.imread(path_each_img)
        img_bgr_rotated = util_rotate(img_bgr)

        # img_bgr_rotated = cv2.detailEnhance(img_bgr_rotated, sigma_s=22, sigma_r=13)

        img_rgb = cv2.cvtColor(img_bgr_rotated, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr_rotated, cv2.COLOR_BGR2GRAY)

        thresh1, img_bi = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # print(thresh1)
        ## plot pixel value distribution
        # sb.displot(img_gray.flatten(), kde=True)

        ## plot result of thresholding
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(img_gray, cmap='bone')
        # plt.subplot(122)
        # plt.imshow(img_bi, cmap='bone')

        # Export all result of thresholding]
        # path_thresh='visualize/thresholding'
        # if pos==0:
        #     if os.path.exists(path_thresh):send2trash.send2trash(path_thresh);os.makedirs(path_thresh)
        #     else:os.makedirs(path_thresh)
        # if pos % 25 == 0:
        #     plt.figure(figsize=(16,9))
        #     plt.suptitle(img_name)
        #     plt.subplot(121)
        #     plt.imshow(img_bgr[:,:,::-1])
        #     plt.subplot(122)
        #     plt.imshow(img_bi, cmap='bone')
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(path_thresh,img_name))
        #     plt.close()
        # continue

        kernel = np.ones((8, 13), np.uint8)
        img_bi_dilate = cv2.dilate(img_bi, kernel)
        # Export all result of dilation
        # path_dilate= 'visualize/dilate'
        # if pos==0:
        #     if os.path.exists(path_dilate):send2trash.send2trash(path_dilate);os.makedirs(path_dilate)
        #     else:os.makedirs(path_dilate)
        # if pos % 25 == 0:
        #     plt.figure(figsize=(16,9))
        #     plt.suptitle(img_name)
        #     plt.subplot(121)
        #     plt.imshow(img_bi,cmap='bone')
        #     plt.subplot(122)
        #     plt.imshow(img_bi_dilate, cmap='bone')
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(path_dilate, img_name))
        #     plt.close()
        # continue

        (contours, _) = cv2.findContours(img_bi_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        coor_14_words = []
        for contour in contours:
            coor_14_words.append(cv2.boundingRect(contour))
        # ocr_extract_text(img_bgr_rotated, coor_14_words,img_name)

        # img_bgr_rect = np.copy(img_bgr_rotated)
        # n_contours = len(contours)
        # if n_contours != 14:
        #     print(f'{img_name} have {n_contours} contours')
        # for contour in contours:
        #     (x, y, w, h) = cv2.boundingRect(contour)
        #     cv2.rectangle(img_bgr_rect, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # # plot specific graph
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(img_bgr[:, :, ::-1])
        # plt.subplot(122)
        # plt.imshow(img_bgr_rect[:, :, ::-1])
        # plt.show()

        ## Export all result of bounding box
        # path_rect = 'visualize/rect_each_word'
        # path_rect_fail = 'visualize/rect_each_word/failed'
        # if pos == 0:
        #     if os.path.exists(path_rect):
        #         send2trash.send2trash(path_rect);os.makedirs(path_rect_fail)
        #     else:
        #         os.makedirs(path_rect_fail)
        # if pos % 25 == 0 or n_contours != 14:
        #     plt.figure(figsize=(16, 9))
        #     plt.suptitle(img_name + '_' + str(n_contours))
        #     plt.subplot(121)
        #     plt.imshow(img_bgr[:, :, ::-1])
        #     plt.subplot(122)
        #     plt.imshow(img_bgr_rect[:, :, ::-1])
        #     plt.tight_layout()
        #     if n_contours == 14:
        #         plt.savefig(os.path.join(path_rect, img_name))
        #     else:
        #         plt.savefig(os.path.join(path_rect_fail, img_name))
        #     plt.close()
        # continue

        border = 13
        h_entire, w_entire, _ = img_bgr_rotated.shape
        for pos_word, (x, y, w, h) in enumerate(coor_14_words[::-1]):
            row_start = np.clip(y - border, 0, h_entire)
            row_stop = np.clip(y + h + border, 0, h_entire)
            col_start = np.clip(x - border, 0, w_entire)
            col_stop = np.clip(x + w + border, 0, w_entire)
            img_each_word = img_bgr_rotated[row_start:row_stop, col_start:col_stop]

            # path_each_word = 'visualize/each_word'
            # if pos == 0:
            #     if os.path.exists(path_each_word):
            #         send2trash.send2trash(path_each_word);
            #         os.makedirs(path_each_word)
            #     else:
            #         os.makedirs(path_each_word)
            # plt.figure(figsize=(16, 9))
            # plt.imshow(img_each_word[:, :, ::-1])
            # plt.title(img_name + '_' + str(count_word))
            # plt.savefig(path_each_word + '/' + img_name[:-4] + '_' + str(count_word) + '.png')
            # plt.close()
            # count_word += 1
            # continue

            text = tess.image_to_string(img_each_word, lang='tha')
            # reader = easyocr.Reader(['th', 'en'], gpu=True)
            # results = reader.readtext(img_each_word)

            # print(count_word)
            # print(results,'\n\n')

            # count_word+=1
            # continue

            # Show cropped image
            # print(f'Img : {img_name} || Word at {count_word} : {text}')
            # plt.imshow(img_each_word[:, :, ::-1])
            # plt.show()

            ## Clean text process

            idx_space = text.find(' ')
            idx_comma = text.find(',')
            idx_fullstop = text.find('.')
            idx_num_list = []
            for pos_char, char in enumerate(text):
                try:
                    if char.isdigit() and not text[pos_char + 1].isdigit():
                        idx_num_list.append(pos_char)
                except:
                    pass
            if idx_space != -1:
                text_cleaned = text[idx_space + 1:]
            elif idx_comma != -1:
                text_cleaned = text[idx_comma + 1:]
            elif idx_fullstop != -1 and idx_fullstop != len(text) - 1:
                text_cleaned = text[idx_fullstop + 1:]
            elif idx_num_list:
                idx = max(idx_num_list)
                text_cleaned = text[idx + 1:]
            else:
                text_cleaned = text

            text_correct = ptn.correct(text_cleaned)

            ## Export text predict to excel
            dict_text['img'].append(img_name + '_' + str(count_word))
            dict_text['true_label'].append(true_label_list[count_word])
            dict_text['pure_text'].append(text)
            dict_text['text_cleaned'].append(text_cleaned)
            dict_text['text_correct'].append(text_correct)

            count_word += 1

    ## Export text to excel
    df_text = pd.DataFrame(dict_text)
    df_text.to_excel('visualize/output.xlsx', index=False)


def main():
    split_each_word()


main()
