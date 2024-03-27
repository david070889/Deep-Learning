import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_img(f):
    f=open(f)
    lines=f.readlines()
    imgs, lab=[], []
    for i in range(len(lines)):
        fn, label = lines[i].split(' ')
        
        im1=cv2.imread(fn)
        im1=cv2.resize(im1, (256,256))
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        
        im1 = cv2.calcHist([im1], [0], None, [256], [0, 256])


        
        vec = np.reshape(im1, [-1])
        imgs.append(vec) 
        lab.append(int(label))
        
    imgs= np.asarray(imgs, np.float32)
    lab= np.asarray(lab, np.int32)
    return imgs, lab 


if __name__ == '__main__':

    x, y = load_img('train.txt')
    tx, ty = load_img('test.txt')
    clf = svm.SVC()
    clf.fit(x, y)
    predictions = clf.predict(tx)
    accuracy = accuracy_score(ty, predictions)
    matrix = classification_report(ty, predictions)
    print(f'Accuracy: {accuracy}')
    print("Classification Report")
    print(matrix)