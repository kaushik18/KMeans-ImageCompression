# Kaushik Nadimpalli
# Assignment 5
# CS6375.002 - Anjum Chida

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io, img_as_float

def nearest_centroids(X,c):
    K = np.size(c,0)
    i_d_x = np.zeros((np.size(X,0),1))
    our_arr = np.empty((np.size(X,0),1))
    for i in range(0,K):
        y = c[i]
        val = np.ones((np.size(X,0),1))*y
        y = np.power(np.subtract(X,val),2)
        x = np.sum(y,axis = 1)
        x = np.asarray(x)
        x.resize((np.size(X,0),1))
        our_arr = np.append(our_arr, x, axis=1)
    our_arr = np.delete(our_arr,0,axis=1)
    i_d_x = np.argmin(our_arr, axis=1)
    return i_d_x

def calculate_centroids(X,i_d_x,K):
    n = np.size(X,1)
    centroids = np.zeros((K,n))
    for i in range(0,K):
        mean_of_cluster = i_d_x
        mean_of_cluster = mean_of_cluster.astype(int)
        total = sum(mean_of_cluster);
        mean_of_cluster.resize((np.size(X,0),1))
        matrixTotal = np.matlib.repmat(mean_of_cluster,1,n)
        mean_of_cluster = np.transpose(mean_of_cluster)
        total = np.multiply(X,matrixTotal)
        centroids[i] = (1/total)*np.sum(total,axis=0)
    return centroids

def k_means_clustering(img_vectors, k, num_iterations):
    img_compression = np.full((img_vectors.shape[0],), -1)
    initial_cluster = np.random.rand(k, 3)
    for i in range(num_iterations):
        print('Iteration No: ' + str(i + 1))
        labeled_pts = [None for k_i in range(k)]
        for img_i, img in enumerate(img_vectors):
            img_row = np.repeat(img, k).reshape(3, k).T
            closest_label = np.argmin(np.linalg.norm(img_row - initial_cluster, axis=1))
            img_compression[img_i] = closest_label
            if (labeled_pts[closest_label] is None):
                labeled_pts[closest_label] = []
            labeled_pts[closest_label].append(img)
        for k_i in range(k):
            if (labeled_pts[k_i] is not None):
                new_cluster_prototype = np.asarray(labeled_pts[k_i]).sum(axis=0) / len(labeled_pts[k_i])
                initial_cluster[k_i] = new_cluster_prototype
    return (img_compression, initial_cluster)

if __name__ == '__main__':
    jpeg_img = sys.argv[1]
    K = int(sys.argv[2])
    no_iterations = int(sys.argv[3])

    image = io.imread(jpeg_img)[:, :, :3]
    image = img_as_float(image)
    image_dimensions = image.shape
    image_name = image
    img_vectors = image.reshape(-1, image.shape[-1])

    img_compression, centroids_colored = k_means_clustering(img_vectors, k=K, num_iterations=no_iterations)
    compressed_image = np.zeros(img_vectors.shape)
    for i in range(compressed_image.shape[0]):
        compressed_image[i] = centroids_colored[img_compression[i]]
    compressed_image = compressed_image.reshape(image_dimensions)

    print('Process - Currently saving the compressed image')
    io.imsave('New_Compressed_Image.jpg' , compressed_image)
    print('Compressed Image Created')
    data = os.stat(jpeg_img)
    print("Input Image Size : ",data.st_size/1024,"KB")
    data = os.stat('New_Compressed_Image.jpg')
    print("Compressed Image Size : ",data.st_size/1024,"KB")
