import cv2

face_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
ind = 0
for i in range(200):
    img = cv2.imread(f"./datasets/none/none_{i}.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    imgNum = 0
    for (x, y, w, h) in faces:
        dl = max(w, h)
        cropped = img[y - int(dl / 4):y + dl + int(dl / 4), x - int(dl / 4):x + dl + int(dl / 4)]
        try:
            # cropped = cv2.resize(cropped, dsize=(640, 640), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f"./img_crop/none/none_face{ind}.jpg", cropped)
            ind += 1
            print(ind)
        except:
            pass
