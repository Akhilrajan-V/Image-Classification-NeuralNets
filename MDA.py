import numpy as np
import matplotlib.pyplot as plt


class MDA:

    def compute_MDA_svm(self, train_imgs, train_labels, test_imgs, val_imgs, pc_count, img_h, img_w, show_img=False):

        train_imgs_copy = train_imgs.reshape(train_imgs.shape[0],-1)
        test_imgs_copy = test_imgs.reshape(test_imgs.shape[0], -1)
        val_imgs_copy = val_imgs.reshape(val_imgs.shape[0], -1)

        mu_hat = {}
        mu = []
        for i in np.unique(train_labels):
            mu_hat[format(i)] = np.mean(train_imgs_copy[train_labels == i], axis=0)
            mu.append(mu_hat[format(i)])

        mu = np.array(mu)
        mu_0 = np.mean(mu, axis=0)

        Sigma_b = []
        for i in np.unique(train_labels):
            Sigma_b.append(mu_hat[format(i)] - mu_0)

        Sigma_b = np.mean(Sigma_b, axis=0)
        Sigma_b = np.matmul(Sigma_b.reshape(img_h*img_w,1),(Sigma_b.reshape(img_h*img_w,1)).T)

        cov_hat = {}
        cov = []
        for i in np.unique(train_labels):
            cov_hat[format(i)] = np.cov(train_imgs_copy[train_labels == i].T)
            cov.append(cov_hat[format(i)])

        cov = np.array(cov)
        cov_0 = np.mean(cov, axis=0)

        Sigma_wb = np.matmul(np.linalg.pinv(cov_0), Sigma_b)
        eigenValues_org, eigenVectors_org = np.linalg.eigh(Sigma_wb)
        sorted_eig = np.argsort(-eigenValues_org)
        eigenValues = eigenValues_org[sorted_eig]
        eigenVectors = eigenVectors_org[:, sorted_eig]
        pc = eigenVectors[:, 0:pc_count]

        P = np.matmul(pc,pc.T)
        reconstruction_train = np.matmul(train_imgs_copy, P)
        reconstruction_test = np.matmul(test_imgs_copy, P)
        reconstruction_val = np.matmul(val_imgs_copy, P)

        if show_img == True:
            for i in range(0,400):
                img = train_imgs_copy[i, :]
                reconstruction_i = reconstruction_train[i].reshape(img_h, img_w)
                plt.imshow(img.reshape(img_h, img_w), cmap='gray')
                plt.show()
                plt.imshow(reconstruction_i, cmap='gray')
                plt.show()

        return reconstruction_train, reconstruction_test, reconstruction_val

    def compute_MDA(self, train_imgs, train_labels, test_imgs, n_components, img_h, img_w, show_img=False):

        train_imgs_copy = train_imgs.reshape(train_imgs.shape[0],-1)
        test_imgs_copy = test_imgs.reshape(test_imgs.shape[0], -1)

        mu_hat = {}
        mu = []
        for i in np.unique(train_labels):
            mu_hat[format(i)] = np.mean(train_imgs_copy[train_labels == i], axis=0)
            mu.append(mu_hat[format(i)])

        mu = np.array(mu)
        mu_0 = np.mean(mu, axis=0)

        Sigma_b = []
        for i in np.unique(train_labels):
            Sigma_b.append(mu_hat[format(i)] - mu_0)

        Sigma_b = np.mean(Sigma_b, axis=0)
        Sigma_b = np.matmul(Sigma_b.reshape(img_h*img_w,1),(Sigma_b.reshape(img_h*img_w,1)).T)

        cov_hat = {}
        cov = []
        for i in np.unique(train_labels):
            cov_hat[format(i)] = np.cov(train_imgs_copy[train_labels == i].T)
            cov.append(cov_hat[format(i)])

        cov = np.array(cov)
        cov_0 = np.mean(cov, axis=0)

        Sigma_wb = np.matmul(np.linalg.pinv(cov_0), Sigma_b)
        eigenValues_org, eigenVectors_org = np.linalg.eigh(Sigma_wb)
        sorted_eig = np.argsort(-eigenValues_org)
        eigenValues = eigenValues_org[sorted_eig]
        eigenVectors = eigenVectors_org[:, sorted_eig]
        pc = eigenVectors[:, 0:n_components]

        P = np.matmul(pc,pc.T)
        reconstruction_train = np.matmul(train_imgs_copy, P)
        reconstruction_test = np.matmul(test_imgs_copy, P)

        if show_img == True:
            for i in range(0,400):
                img = train_imgs_copy[i, :]
                reconstruction_i = reconstruction_train[i].reshape(img_h, img_w)
                plt.imshow(img.reshape(img_h, img_w), cmap='gray')
                plt.show()
                plt.imshow(reconstruction_i, cmap='gray')
                plt.show()

        return reconstruction_train, reconstruction_test
