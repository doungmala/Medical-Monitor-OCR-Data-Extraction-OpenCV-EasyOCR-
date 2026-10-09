import json
import os
import sys
import cv2
import numpy as np
import imutils
from collections import Counter
import pytesseract
import re
import contextlib
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Debug flag
DEBUG_MODE = False

def is_number_only(text):
    """ตรวจสอบว่าข้อความเป็นตัวเลขหรือไม่"""
    if not text or not isinstance(text, str):
        return False
    cleaned = re.sub(r'[^0-9.\-]', '', text).strip()
    cleaned = re.sub(r'\.{2,}', '.', cleaned)
    cleaned = re.sub(r'(?<=.)-', '', cleaned)
    try:
        float(cleaned)
        return True
    except:
        return False

def safe_readtext(reader, image_path):
    """อ่าน OCR โดยตรวจสอบก่อนว่าไฟล์มีอยู่และไม่ว่าง"""
    if not os.path.exists(image_path):
        if DEBUG_MODE:
            print(f"[WARN] File not found: {image_path}")
        return []
    if os.path.getsize(image_path) == 0:
        if DEBUG_MODE:
            print(f"[WARN] Empty file: {image_path}")
        return []
    return reader.readtext(image_path)

# โหลด easyocr แบบ suppress log
with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        import easyocr
reader = easyocr.Reader(['en'], verbose=False)

output_dir = r"C:/back-end/ocr_crop"
os.makedirs(output_dir, exist_ok=True)

# อ่านพารามิเตอร์จาก command line
image_paths = sys.argv[1:]
if len(image_paths) < 2:
    pathA = os.path.join(output_dir, "testA.jpg")
    pathB = os.path.join(output_dir, "testB.jpg")
    print("[INFO] Using default image paths for testing")
else:
    pathA, pathB = image_paths[0], image_paths[1]

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method in ["right-to-left", "bottom-to-top"]:
        reverse = True
    if method in ["top-to-bottom", "bottom-to-top"]:
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i],
                                        reverse=reverse))
    return (cnts, boundingBoxes)

# ====== ส่วนการประมวลผลภาพคงโครงสร้างเดิมแต่เปลี่ยน path และ readtext ======

num_realvpn=0
num_realtmp=0
num_realvolume=0
num_realufgoal=0
num_realufrate=0
num_realflw=0

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
    stringname = os.path.join(output_dir, f'last{count}.jpg')
    x, y, w, h = cv2.boundingRect(c)
    if count == 3:
        last2 = image[y-10:y+h+10, x-10:x+w+100]
    elif count == 2:
        last2 = image[y:y+h, x-30:x+w]
    else:
        last2 = image[y:y+h, x:x+w]
    cv2.imwrite(stringname, last2)

# ven
resultsvpn = safe_readtext(reader, os.path.join(output_dir, 'last1.jpg'))
if resultsvpn:
    raw = resultsvpn[0][1]
    if is_number_only(raw):
        num_realvpn = raw.replace("|", "1").replace(",", ".")
    else:
        num_realvpn = "0"
else:
    num_realvpn = "0"

# tmp
imagetmp = cv2.imread(os.path.join(output_dir, 'last2.jpg'))
gray = cv2.cvtColor(imagetmp, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
openingtmp = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
kernel = np.ones((3, 3), np.uint8)
img_erosiontmp = cv2.erode(openingtmp, kernel, iterations=1)
if DEBUG_MODE:
    cv2.imshow('img_erosiontmp', img_erosiontmp)

contoursBLUnit = cv2.findContours(img_erosiontmp.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoursBLUnit = imutils.grab_contours(contoursBLUnit)
cnts, _ = sort_contours(contoursBLUnit, method="left-to-right")
countsub = 0
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    lasttmp = img_erosiontmp[y:y+h, x:x+w]
    if DEBUG_MODE:
        cv2.imshow('lasttmp', lasttmp)
    contoursBLUnitlasttmp = cv2.findContours(lasttmp.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursBLUnitlasttmp = imutils.grab_contours(contoursBLUnitlasttmp)
    contoursBLUnitlasttmp = sorted(contoursBLUnitlasttmp, key=cv2.contourArea, reverse=True)[:1]
    for c2 in contoursBLUnitlasttmp:
        xlasttmp, ylasttmp, wlasttmp, hlasttmp = cv2.boundingRect(c2)
        lasttmp2 = lasttmp[ylasttmp:ylasttmp+hlasttmp, xlasttmp:xlasttmp+wlasttmp]
        if DEBUG_MODE:
            cv2.imshow('lasttmp2', lasttmp2)
            cv2.waitKey(0)
        if hlasttmp <= 53 and wlasttmp >= 23 and wlasttmp <= 33:
            countsub += 1

resultstmp = safe_readtext(reader, os.path.join(output_dir, 'last2.jpg'))
for (bboxtmp, texttmp, probtmp) in resultstmp:
    num_realtmp = texttmp.replace("|", "1")
    if not num_realtmp.isnumeric() or countsub >= 1:
        if len(num_realtmp) == 1:
            num_realtmp = '-' + num_realtmp[0]
        elif len(num_realtmp) == 2:
            num_realtmp = '-' + num_realtmp[1]
        elif len(num_realtmp) == 3:
            num_realtmp = '-' + num_realtmp[1] + num_realtmp[2]
        elif len(num_realtmp) == 4:
            num_realtmp = '-' + num_realtmp[1] + num_realtmp[2] + num_realtmp[3]
        break

# volume
imagevolume = cv2.imread(os.path.join(output_dir, 'last3.jpg'))
gray = cv2.cvtColor(imagevolume, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
openingven = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
kernel = np.ones((3, 3), np.uint8)
img_erosion = cv2.erode(openingven, kernel, iterations=1)
cv2.imwrite(os.path.join(output_dir, 'volume2.jpg'), img_erosion)
results = safe_readtext(reader, os.path.join(output_dir, 'volume2.jpg'))
if results:
    raw = results[0][1]
    if is_number_only(raw):
        num_realvolume = raw.replace("|", "1").replace(",", ".")
    else:
        num_realvolume = "0"
else:
    num_realvolume = "0"

sentence = num_realvolume
if "." not in sentence:
    resvolume = ''.join([char for char in num_realvolume if char.isalnum()])
    if len(resvolume) == 3:
        num_realvolume = f"{resvolume[0]}.{resvolume[1]}{resvolume[2]}"
    elif len(resvolume) == 2:
        num_realvolume = f"{resvolume[0]}.{resvolume[1]}"

# uf goal
resultsufgoal = safe_readtext(reader, os.path.join(output_dir, 'last5.jpg'))
if not resultsufgoal:
    num_realufgoal = "0"
else:
    for (_, textufgoal, _) in resultsufgoal:
        if is_number_only(textufgoal):
            num_realufgoal = textufgoal.replace("|", "1").replace(",", ".")
        else:
            num_realufgoal = "0"
        break
if "." not in num_realufgoal:
    resufgoal = ''.join([char for char in num_realufgoal if char.isalnum()])
    if len(resufgoal) == 3:
        num_realufgoal = f"{resufgoal[0]}.{resufgoal[1]}{resufgoal[2]}"
    elif len(resufgoal) == 2:
        num_realufgoal = f"{resufgoal[0]}.{resufgoal[1]}"

# uf rate
imagerate = cv2.imread(os.path.join(output_dir, 'last6.jpg'))
grayrate = cv2.cvtColor(imagerate, cv2.COLOR_BGR2GRAY)
_, threshrate = cv2.threshold(grayrate, 180, 255, cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
openingrate = cv2.morphologyEx(threshrate, cv2.MORPH_OPEN, kernel)
kernel = np.ones((3, 3), np.uint8)
erosionrate = cv2.erode(openingrate, kernel, iterations=1)
erosionrate = ~erosionrate
cv2.imwrite(os.path.join(output_dir, 'ufrate_rep.jpg'), erosionrate)
results = safe_readtext(reader, os.path.join(output_dir, 'ufrate_rep.jpg'))
for (bbox, text, prob) in results:
    num_realufrate = text.replace("|", "1").replace(",", ".")
    if "." not in num_realufrate:
        resufrate = ''.join([char for char in num_realufrate if char.isalnum()])
        if len(resufrate) == 3:
            num_realufrate = f"{resufrate[0]}.{resufrate[1]}{resufrate[2]}"
        elif len(resufrate) == 2:
            num_realufrate = f"{resufrate[0]}.{resufrate[1]}"
        else:
            num_realufrate = "0"

# flw
imageflw = cv2.imread(pathB)
gray = cv2.cvtColor(imageflw, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
thresh = ~thresh
kernel = np.ones((2, 2), np.uint8)
openingven = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
kernel = np.ones((1, 5), np.uint8)
img_erosion = cv2.erode(thresh, kernel, iterations=1)
cv2.imwrite(os.path.join(output_dir, 'flw.jpg'), img_erosion)
results = safe_readtext(reader, os.path.join(output_dir, 'flw.jpg'))
for (_, text, _) in results:
    num_realflw = "".join(filter(str.isdigit, text.replace("o", "0").replace("O", "0")))
    break

# output
data = f"{num_realvpn},{num_realtmp},{num_realvolume},{num_realufgoal},{num_realufrate},{num_realflw}"
with open("my_text_file.txt", 'w') as file:
    file.write(data)

result_json = {
    "ven_press": num_realvpn,
    "tmp": num_realtmp,
    "uf_volume": num_realvolume,
    "uf_goal": num_realufgoal,
    "uf_rate": num_realufrate,
    "bflw": num_realflw
}
print(json.dumps(result_json, indent=2, ensure_ascii=False))
