import os
import time
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as im
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay
from tqdm import tqdm
from scipy.special import expit

ROOT_FOLDER_PATH = "C:/Users/Zalamaan/Documents/Repos/NICU_Data/DepthFrameFullPrec_prePT/"
# allPts = [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, *range(21, 35), *range(90, 95)]
allPts = [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, *range(21, 35)]

def testImshow(img):
    while True:
        cv2.imshow("test", img)

        # If user presses ESCAPE or clicks the close button, end    
        key = cv2.waitKey(1)
        if (key == 27) or (cv2.getWindowProperty("test", cv2.WND_PROP_VISIBLE) != 1):
            cv2.destroyAllWindows()
            break

def getImageMSE(img, baselineImg):
    mse = ((img - baselineImg)**2).mean()
    return mse

def getImgIDProb(mse, threshold=0.5):
    return 0.5

def findUniqueFileNames(rootFolderPath, patientId):
    nurseFiles_path = rootFolderPath + 'p' + str(patientId) + '/nurse/*_0.png'
    nurseFiles = glob(nurseFiles_path)
    nooneFiles_path = rootFolderPath + 'p' + str(patientId) + '/noone/*_0.png'
    nooneFiles = glob(nooneFiles_path)

    uniqueFiles = list(set(nooneFiles).symmetric_difference(set(nurseFiles)))

    uniqueFileNames = []
    for file in uniqueFiles:
        uniqueFileNames.append(os.path.basename(file).split('.')[0][:-2])

    return uniqueFileNames

print('Started')
patientList = []
fileNames = []
baselineImgs = []
# allPts = [1]
for pt in allPts:
    uniqueFileNames = findUniqueFileNames(ROOT_FOLDER_PATH, pt)

    for fileName in uniqueFileNames:
        imgPath = ROOT_FOLDER_PATH + 'p' + str(pt) + '/noone/' + fileName + '*.png'
        imgPath = glob(imgPath)
        print(imgPath)
        if imgPath == []:
            continue
        imgPath = imgPath[0]
        img = cv2.imread(imgPath, -1)
        baselineImg = img.astype(np.float32)
        patientList.append(pt)
        fileNames.append(fileName)
        baselineImgs.append(baselineImg)

all_MSEs = []
all_y_true = []
for pt, fileName, baselineImg in zip(patientList, fileNames, baselineImgs):
        
    imgPaths_nurse = ROOT_FOLDER_PATH + 'p' + str(pt) + '/nurse/' + fileName + '*.png'
    imgPaths_noone = ROOT_FOLDER_PATH + 'p' + str(pt) + '/noone/' + fileName + '*.png'

    imgList_nurse = glob(imgPaths_nurse)
    imgList_noone = glob(imgPaths_noone)

    firstRun = True
    for imgPath in tqdm(imgList_nurse):
        img = cv2.imread(imgPath, -1)
        img = cv2.resize(img, (baselineImg.shape[1], baselineImg.shape[0]))
        if firstRun:
            nurseImgs = img.reshape((1, img.shape[0], img.shape[1]))
            firstRun = False
        else:
            nurseImgs = np.concatenate((nurseImgs, [img]))

    nurseMSEs = ((nurseImgs-baselineImg)**2).mean(axis=(1, 2))
    y_true = np.ones(nurseMSEs.shape[0])

    firstRun = True
    for imgPath in tqdm(imgList_noone):
        img = cv2.imread(imgPath, -1)
        img = cv2.resize(img, (baselineImg.shape[1], baselineImg.shape[0]))
        if firstRun:
            nooneImgs = img.reshape((1, img.shape[0], img.shape[1]))
            firstRun = False
        else:
            nooneImgs = np.concatenate((nooneImgs, [img]))

    nooneMSEs = ((nooneImgs-baselineImg)**2).mean(axis=(1, 2))
    
    MSEs = np.concatenate((nurseMSEs, nooneMSEs))
    y_true = np.concatenate((y_true, np.zeros(nooneMSEs.shape[0])))
    print(f'MSE shape: {MSEs.shape}, y_true shape: {y_true.shape}')

    all_MSEs.append(MSEs)
    all_y_true.append(y_true)

MSEs = np.concatenate(all_MSEs).reshape(-1, 1)
y_true = np.concatenate(all_y_true)
print(f'MSE shape: {MSEs.shape}, y_true shape: {y_true.shape}')

X = MSEs.copy()
y = y_true.copy()

cv = StratifiedKFold(n_splits=5)
classifier = LogisticRegression(random_state=0)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])

    # plt.figure(1, figsize=(4, 3))
    # plt.clf()
    # plt.scatter(X.ravel(), y, color="black", zorder=20)
    # X_test = np.linspace(-5, 10, 300)

    # loss = expit(X_test * classifier.coef_ + classifier.intercept_).ravel()
    # plt.plot(X_test, loss, color="red", linewidth=3)

    # plt.ylabel("y")
    # plt.xlabel("X")
    # plt.xticks(range(-5, 10))
    # plt.yticks([0, 0.5, 1])
    # plt.ylim(-0.25, 1.25)
    # plt.xlim(-4, 10)
    # plt.legend(
    #     "Linear Regression Model",
    #     loc="lower right",
    #     fontsize="small",
    # )
    # plt.tight_layout()
    # plt.show()

    viz = RocCurveDisplay.from_estimator(
        classifier,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic example",
)
ax.legend(loc="lower right")
plt.show()
