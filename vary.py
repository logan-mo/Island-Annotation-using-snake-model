import cv2
import numpy as np
import pandas as pd

# Change numpy's output format
np.set_printoptions(formatter={'float_kind':"{:.2f}".format})

def funcRotate(degree=0):
    degree = cv2.getTrackbarPos('degree','Frame')
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    rotated_image = cv2.warpAffine(three_D, rotation_matrix, (width, height))
    cv2.imshow('Frame', rotated_image)


if __name__ == '__main__':
    data = pd.read_csv('island_annotation.csv').drop(columns=['image_path']).values.transpose()
    
    C = np.cov(data)

    # Covariance matrix can come out as complex numbers with very small (0) imaginary value
    # So, make it real
    C = np.abs(C)

    print('Covariance Matrix:\n', C)
    print('Shape:', C.shape)
    print()

    vals, vecs = np.linalg.eig(C)

    # vals and vecs can come out as complex numbers with very small (0) imaginary value
    # So, make it real
    vals = np.abs(vals)
    vecs = np.abs(vecs)

    print('Eigenvalues:')
    print(vals)
    print('Shape:', vals.shape)
    print()
    print('Eigenvectors:')
    print(vecs)
    print('Shape:', vecs.shape)

    # Mean face
    mean_face = data.mean(axis=1)
    print()
    print('Mean face:')
    print(mean_face)
    print('Shape:', mean_face.shape)
    
    mean_features = data.mean(axis=1)
    mean_snake = mean_features.reshape((-1,2))
    
    original = np.zeros((700, 700))
    for i in range(0, mean_snake.shape[0]):
        original[int(mean_snake[i, 0]), int(mean_snake[i, 1])] = 255
    
    kernel = np.ones((5, 5), np.uint8)
    original = cv2.dilate(original, kernel, iterations=1)
    
    three_D = np.zeros((700, 700, 3))
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            if(original[i, j] == 255):
                three_D[i, j] = (255, 255, 255)
    
    
    three_D[np.all(three_D == (255, 255, 255), axis=-1)] = (255,0,0)
    
    
    
    height, width = three_D.shape[:2]
    cv2.namedWindow('Frame')
    degree=0
    cv2.createTrackbar('degree','Frame',degree,360,funcRotate)
    funcRotate(0)
    cv2.imshow('Frame',three_D)
    cv2.waitKey(0)
