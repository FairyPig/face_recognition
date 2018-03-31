import numpy as np
import os, math, os.path, glob, random, cv2
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import scipy.io as sio
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq

class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / np.max(img))
        self.img = img * 255
        
        #cv2.imshow('sqrt', self.img)
        #cv2.waitKey(0)
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        #ssert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self):
        height, width = self.img.shape
        # Compute gradient(magnitude & angle) of each pixels 
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        # Cell gradient, each contains bin_size * bin_size pixels
        cell_gradient_vector = np.zeros((height / self.cell_size, width / self.cell_size, self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                # Compute cell gradient (a vector of 8)
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        # Render gradient to image
        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        # Each block contains 4 cells
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers
 
    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

# img = cv2.imread('1.pgm', cv2.IMREAD_GRAYSCALE)
# hog = Hog_descriptor(img, cell_size=4, bin_size=8)
# vector, image = hog.extract()
# print(np.array(vector).shape)
# plt.imshow(image, cmap=plt.cm.gray)
# plt.show()

# HOG特征维度（最后输出的是一个 594 * 32 的向量）
# (28 - 1)*(23 - 1) * (4 * 8) = 19008
# 原始维度
# 112 * 92 = 10948

def hog2vec(filename):
    # Read as gray image
    img = cv2.imread(filename, 0)
    hog = Hog_descriptor(img, cell_size = 4, bin_size=8)
    vector, image = hog.extract()
    rows, cols = np.array(vector).shape
    hogVec = np.zeros((1, rows*cols))
    hogVec = np.reshape(vector, (1, rows*cols))
    return hogVec

# 加载图像
def loadImageSet(dataSetDir):
    train_face_hog = np.mat(np.zeros((40*8, 594*32)))
    train_face_number = np.zeros(40*8)
    test_face_hog = np.mat(np.zeros((40*2, 594*32)))
    test_face_number = np.zeros(40*2)

    for i in range(40):
        people_num = i + 1
        count = 0
        for j in range(10):
            face_num = j + 1
            filename = dataSetDir + '/s' + str(people_num) + '/' + str(face_num) + '.bmp'
            
            hogVec = hog2vec(filename)
            if face_num == 4:
                test_face_hog[2*i, :] = hogVec
                test_face_number[2*i] = people_num
            elif  face_num == 8:
                test_face_hog[2 * i + 1, :] = hogVec
                test_face_number[2 * i + 1] = people_num
            else:
                train_face_hog[i*8 + count, :] = hogVec
                train_face_number[i*8+count] = people_num
                count += 1
    
    return train_face_hog, train_face_number, test_face_hog, test_face_number

def runHOG():
    cell_size = 16
    bin_size = 9
    
    # Load dataset
    mat_file05 = 'C:/Users/WuJiqiang/Desktop/patternClassification/database/PIE/Pose05_64x64'
    mat_file07 = 'C:/Users/WuJiqiang/Desktop/patternClassification/database/PIE/Pose07_64x64'
    mat_file09 = 'C:/Users/WuJiqiang/Desktop/patternClassification/database/PIE/Pose09_64x64'
    mat_file27 = 'C:/Users/WuJiqiang/Desktop/patternClassification/database/PIE/Pose27_64x64'
    mat_file29 = 'C:/Users/WuJiqiang/Desktop/patternClassification/database/PIE/Pose29_64x64'

    data_pose05 = sio.loadmat(mat_file05)
    data_pose07 = sio.loadmat(mat_file07)
    data_pose09 = sio.loadmat(mat_file09)
    data_pose27 = sio.loadmat(mat_file27)
    data_pose29 = sio.loadmat(mat_file29)

    train_face_hog = np.mat(np.zeros((2992+1425+1428+2989+1428, (int)(64/cell_size - 1)*(64/cell_size - 1)* bin_size * 4)))
    test_face_hog = np.mat(np.zeros((340+204+204+340+204,  (int)(64/cell_size - 1)*(64/cell_size - 1)* bin_size * 4)))
    train_face_number = [0 for x in range(2992+1425+1428+2989+1428)]
    test_face_number = [0 for x in range(340+204+204+340+204)]

    m = 0
    n = 0
    img = np.zeros((64, 64))

    for k in range(5):
        if k == 0:
            temp = data_pose05
        elif k == 1:
            temp = data_pose07
        elif k == 2:
            temp = data_pose09
        elif k == 3:
            temp = data_pose27
        else:
            temp = data_pose29
        for i in range(np.shape(temp['isTest'])[0]):

            # 转换为HOG特征提取所需格式
            for u in range(64):
                for v in range(64): 
                    img[u][v] = temp['fea'][i][u*64 + v]
            hog = Hog_descriptor(img, cell_size=cell_size, bin_size=bin_size)
            vector,image = hog.extract()
            rows, cols = np.array(vector).shape
            hogVec = np.zeros((1, rows*cols))
            hogVec = np.reshape(vector, (1, rows*cols))

            if temp['isTest'][i] == 0:
                train_face_hog[m] = hogVec
                train_face_number[m] = temp['gnd'][i]
                m += 1
            else:
                test_face_hog[n] = hogVec
                test_face_number[n] = temp['gnd'][i]
                n += 1

    train_face_number = [int(i) for i in train_face_number]
    test_face_number = [int(i) for i in test_face_number]

    # print(FaceMat)
    # print(judgeImgs)

    n_classes = len(np.unique(train_face_number))
    target_names = []
    for i in range(68):
        names = "person" + str(i)
        target_names.append(names)

    # PCA过程
    n_components = 150
    x = []
    acc_array = []
    maxAcc = 0
    for n_components in range(10,320,10):
        
        print("Extracting the top %d eigenfaces from %d faces"% (n_components, train_face_hog.shape[0]))
        pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(train_face_hog)
        #eigenfaces = pca.components_.reshape((n_components, 28, 256))
        X_train_pca = pca.transform(train_face_hog)
        X_test_pca = pca.transform(test_face_hog)

        # Train a SVM classfication model
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        clf = clf.fit(X_train_pca, train_face_number)
        # 不用PCA
        #clf = clf.fit(train_face_hog, train_face_number)

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        y_pred = clf.predict(X_test_pca)
        count = 0
        for i in range(340+204+204+340+204):
            if y_pred[i] == test_face_number[i]:
                count += 1
        print(count)
        print("正确率: %.4f" % (count / (340.0+204.0+204.0+340.0+204.0)))
        x.append(n_components)
        acc_array.append(count / (340.0+204.0+204.0+340.0+204.0))
        if maxAcc < (count / (340.0+204.0+204.0+340.0+204.0)):
            maxAcc = (count / (340.0+204.0+204.0+340.0+204.0))

    print("maxAcc: %f" % maxAcc)
    plt.plot(x, acc_array, color = 'b')
    plt.xlabel('PCA dimension')
    plt.ylabel('accuracy')
    #plt.title('HOG_PCA_SVM_PIE')
    plt.show()
        #print(classification_report(test_face_number, y_pred, target_names=target_names))
        #print(confusion_matrix(test_face_number, y_pred, labels=range(n_classes)))


    # # print(np.array(train_face_hog).shape)
    # # print(np.array(test_face_hog).shape)

    # # print(train_face_number)
    # # print(test_face_number)
    # # print('~')
    
    # #HOGoperator = np.mat(np.zeros(40*8, vector_size))


if __name__ == '__main__':
    runHOG()