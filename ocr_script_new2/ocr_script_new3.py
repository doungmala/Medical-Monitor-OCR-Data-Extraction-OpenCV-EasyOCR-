import json
import os
import sys
import cv2
import numpy as np
import imutils
from collections import Counter
import pytesseract
import re
import subprocess
import contextlib
import io 
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
##pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def is_number_only(text):
    cleaned = re.sub(r'[^0-9\.\-]', '', text)
    try:
        float(cleaned)
        return True
    except:
        return False


with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        import easyocr
reader = easyocr.Reader(['en'], verbose=False)



output_dir = r"C:/back-end/ocr_crop"
os.makedirs(output_dir, exist_ok=True)  

num_realvpn=0
num_realtmp=0
num_realvolume=0
num_realufgoal=0
num_realufrate=0
num_realflw=0

image_paths = sys.argv[1:]
if len(image_paths) < 2:
    print("กรุณาระบุ path ของภาพ 2 ไฟล์")
    sys.exit(1)

pathA = image_paths[0]
pathB = image_paths[1]

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

#pathA = '2_1.png'
#pathB = '7_2.png'

image = cv2.imread(pathA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
openingven = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

kernel = np.ones((1, 7), np.uint8)
dilated_image = cv2.dilate(openingven, kernel, iterations=6)

contoursBLUnit = cv2.findContours(dilated_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoursBLUnit = imutils.grab_contours(contoursBLUnit)
contoursBLUnit = sorted(contoursBLUnit, key=cv2.contourArea, reverse=True)[:6]
cnts, _ = sort_contours(contoursBLUnit, method="left-to-right")
count = 0
for c in cnts:
    count += 1
    stringname = os.path.join('C:/back-end/ocr_crop/', f'last{count}.jpg')
    x, y, w, h = cv2.boundingRect(c)

    if count == 3:
        last2 = image[y-10:y+h+10, x-10:x+w+100]
    
    elif count==2:
        last2 = image[y:y+h, x-30:x+w]
    else:
        last2 = image[y:y+h, x:x+w]

    cv2.imwrite(stringname, last2)


################# ven #####################################
######################################################เเก้ไขเเล้ว 
resultsvpn = reader.readtext('C:/back-end/ocr_crop/last1.jpg')
if resultsvpn:
    raw = resultsvpn[0][1]
    if is_number_only(raw):
        num_realvpn = raw.replace("|", "1").replace(",", ".")
    else:
        num_realvpn = "0"
else:
    num_realvpn = "0"
#print('num_realvpn :', num_realvpn)


################# tmp #####################################
######################################################เเก้ไขเเล้ว

imagetmp = cv2.imread('C:/back-end/ocr_crop/last2.jpg') 
gray = cv2.cvtColor(imagetmp, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
openingtmp = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
kernel = np.ones((3, 3), np.uint8)
img_erosiontmp = cv2.erode(openingtmp, kernel, iterations=1)

contoursBLUnit = cv2.findContours(img_erosiontmp.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoursBLUnit = imutils.grab_contours(contoursBLUnit)
cnts, _ = sort_contours(contoursBLUnit, method="left-to-right")
countsub=0
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    lasttmp = img_erosiontmp[y:y+h, x:x+w]
    if h <=100:
        countsub=countsub+1
        
        
resultstmp = reader.readtext('C:/back-end/ocr_crop/last2.jpg')
for (bboxtmp, texttmp, probtmp) in resultstmp:
    num_realtmp=texttmp
    
    num_realtmp = num_realtmp.replace("|", "1")
    
    if num_realtmp.isnumeric() == False or countsub >= 1:
        
        if len(num_realtmp) == 1:
            num_realtmp = '-' + num_realtmp[0]
            
        if len(num_realtmp) == 2:
            num_realtmp = '-' + num_realtmp[1]
            
        if len(num_realtmp) == 3:
            num_realtmp = '-' + num_realtmp[1] + num_realtmp[2]
        
        if len(num_realtmp) == 4:
            num_realtmp = '-' + num_realtmp[1] + num_realtmp[2] + num_realtmp[3]
        
    
    if num_realtmp[0] == '4':
        num_realtmp = num_realtmp.replace("4", "-")
    else:
        num_realtmp = num_realtmp
    #print('num_realtmp : ', num_realtmp)
    
    break

#################### volume ##################################
######################################################เเก้เเก้ไขเเล้ว
imagevolume = cv2.imread('C:/back-end/ocr_crop/last3.jpg')
gray = cv2.cvtColor(imagevolume, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
openingven = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
kernel = np.ones((3, 3), np.uint8)
img_erosion = cv2.erode(openingven, kernel, iterations=1)

new_width = 100
new_height = 100
resized_img = cv2.resize(img_erosion, (new_width, new_height))

cv2.imwrite('C:/back-end/ocr_crop/volume2.jpg', img_erosion)

results = reader.readtext('C:/back-end/ocr_crop/volume2.jpg')
if results:
    raw = results[0][1]
    if is_number_only(raw):
        num_realvolume = raw.replace("|", "1").replace(",", ".")
    else:
        num_realvolume = "0"
else:
    num_realvolume = "0"

sentence = num_realvolume
substring = "."

if substring in sentence:
    num_realvolume=num_realvolume
else:
    resvolume = ''.join([char for char in num_realvolume if char.isalnum()])
    resvolumess = len(resvolume)

    if resvolumess == 3:
        num_realvolume = f"{resvolume[0]}.{resvolume[1]}{resvolume[2]}"
    elif resvolumess == 2:
        num_realvolume = f"{resvolume[0]}.{resvolume[1]}"
            
#print("num_realvolume", num_realvolume)


###################### uf goal ################################
######################################################เเก้ไขเเล้ว
resultsufgoal = reader.readtext('C:/back-end/ocr_crop/last5.jpg')
if not resultsufgoal:
    num_realufgoal = "0"
else:
    for (_, textufgoal, _) in resultsufgoal:
        if is_number_only(textufgoal):
            num_realufgoal = textufgoal.replace("|", "1").replace(",", ".")
        else:
            num_realufgoal = "0"
        #print('num_realufgoal :', num_realufgoal)
        break

sentence = num_realufgoal
substring = "."

if substring in sentence:
    num_realufgoal=num_realufgoal
else:
    resufgoal = ''.join([char for char in num_realufgoal if char.isalnum()])
    resufgoalss = len(resufgoal)

    if resufgoalss == 3:
        num_realufgoal = f"{resufgoal[0]}.{resufgoal[1]}{resufgoal[2]}"
    elif resufgoalss == 2:
        num_realufgoal = f"{resufgoal[0]}.{resufgoal[1]}"
        
##################### uf rate #####################
imagerate = cv2.imread('C:/back-end/ocr_crop/last6.jpg') 
grayrate = cv2.cvtColor(imagerate, cv2.COLOR_BGR2GRAY)
_, threshrate = cv2.threshold(grayrate, 180, 255, cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
openingrate = cv2.morphologyEx(threshrate, cv2.MORPH_OPEN, kernel)

kernel = np.ones((3, 3), np.uint8)
erosionrate = cv2.erode(openingrate, kernel, iterations=1)
erosionrate=~erosionrate

cv2.imwrite('C:/back-end/ocr_crop/ufrate_rep.jpg',erosionrate)
#import easyocr
#reader = easyocr.Reader(['en']) # 'en' for English, specify other languages as needed
results = reader.readtext('C:/back-end/ocr_crop/ufrate_rep.jpg')
for (bbox, text, prob) in results:
    num_realufrate=text
    
    
    sentence = num_realufrate
    new_sentence = sentence.replace("|", "1")
    #print(new_sentence)
                
    sentence = new_sentence
    num_realufrate = sentence.replace(",", ".")
    #print(new_sentence)
    
    
    sentence = new_sentence
    substring = "."

    if substring in sentence:
        num_realufrate=new_sentence
        #print(f"'{substring}' is found in the sentence.")
        #print('num_realufgoal : ', num_realufgoal)
    else:
        #print(f"'{substring}' is not found in the sentence.")
        resufrate = ''.join([char for char in num_realufrate if char.isalnum()])
        resufratess = len(resufrate)

        if resufratess == 3:
            num_realufrate = f"{resufrate[0]}.{resufrate[1]}{resufrate[2]}"
        elif resufratess == 2:
            num_realufrate = f"{resufrate[0]}.{resufrate[1]}"
        else:
            num_realufrate=0


################### flw ###################################
###################################################### 

imageflw = cv2.imread(pathB) 

gray = cv2.cvtColor(imageflw, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
thresh=~thresh
kernel = np.ones((2, 2), np.uint8)
openingven = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


kernel = np.ones((1, 5), np.uint8)
img_erosion = cv2.erode(thresh, kernel, iterations=1)

cv2.imwrite('C:/back-end/ocr_crop/flw.jpg',img_erosion)

#import easyocr
#reader = easyocr.Reader(['en']) # 'en' for English, specify other languages as needed
results = reader.readtext('C:/back-end/ocr_crop/flw.jpg')
for (bbox, text, prob) in results:
    num_realflw=text
    
    sentence = num_realflw
    num_realflw = sentence.replace("o", "0").replace("O", "0") #4oo
    
    sentence = num_realflw
    # filter(str.isdigit, sentence) creates an iterator that yields only digit characters.
    # ''.join() concatenates these characters into a single string.
    num_realflw = "".join(filter(str.isdigit, sentence))
    
    #print('num_realflw: ', num_realflw)
    
                
    break
    

r'''
gray = cv2.cvtColor(imageflw, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
thresh = ~thresh 
kernel = np.ones((2, 2), np.uint8)
openingven = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

kernel = np.ones((1, 5), np.uint8)
img_erosion = cv2.erode(thresh, kernel, iterations=1)

cv2.imwrite('flw.jpg',img_erosion)

flw_path = os.path.join(output_dir, 'flw.jpg')
cv2.imwrite(flw_path, openingven)
results = reader.readtext(flw_path)

num_realflw = "0"
for (_, text, _) in results:
    cleaned = (
        text.strip()
        .replace("|", "1")
        .replace("o", "0")
        .replace("O", "0")
        .replace(",", ".")
    )
    
    
    if re.search(r"[a-zA-Z]", cleaned):
        num_realflw = "0"
        break
    
    if re.fullmatch(r"[0-9]+(\.[0-9]+)?", cleaned): 
        num_realflw = cleaned
        break
'''
################## TEXT FILE ####################################
data = f"{num_realvpn},{num_realtmp},{num_realvolume},{num_realufgoal},{num_realufrate},{num_realflw}"
file_name = "my_text_file.txt"
with open(file_name, 'w') as file:
    file.write(data)
#print(f"Content written to {file_name} successfully.")

result_json = {
    "ven_press": num_realvpn,
    "tmp": num_realtmp,
    "uf_volume": num_realvolume,
    "uf_goal": num_realufgoal,
    "uf_rate": num_realufrate,
    "bflw": num_realflw
}


#print("\n OCR RESULT JSON ")
print(json.dumps(result_json, indent=2, ensure_ascii=False))


#cv2.waitKey(0)
#cv2.destroyAllWindows()


