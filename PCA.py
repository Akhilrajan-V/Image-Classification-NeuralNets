import numpy as np


class PCA:

    def __init__(self):

        self.compressed_data = []
        self.projected_data = None
        self.projection_matrix = None

    def __int__(self, X, og_dim):
        # self.X = [img_data[0] for img_data in train_data['train']]
        self.X = X
        self.X = np.array(self.X)
        self.X_og_dim = og_dim
        # self.resize = np.shape(self.X)
        # self.X = self.X.reshape(self.resize[0], self.resize[1]*self.resize[2])
        self._cov_matrix = np.cov(self.X.T)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self._cov_matrix)
        self.eigen_pair = [(np.abs(self.eigenvalues[i]), self.eigenvectors[:, i]) for i in range(len(self.eigenvalues))]

        # Sort the pairs according to decreasing eigenvalues
        self.eigen_pair.sort(key=lambda x: x[0], reverse=True)

        # Percentage of variance to keep
        self.keep_variance = 0.99
        self.required_dim = self._required_dimensions()
        # self.required_dim = 252
        # self.components = np.array([eigvect[1] for eigvect in self.eigen_pair[:self.required_dim]])
        self.transform(self.X)

    def _required_dimensions(self):
        required_variance = self.keep_variance * sum(self.eigenvalues)
        req_dim = 0
        variance = 0
        for i in range(len(self.eigen_pair)):
            variance += self.eigen_pair[i][0]
            if variance >= required_variance:
                req_dim = i + 1
                return req_dim

    def transform(self, X):
        self.projection_matrix = np.empty(shape=(X.shape[1], self.required_dim))

        for idx in range(self.required_dim):
            eigenvector = self.eigen_pair[idx][1]
            self.projection_matrix[:, idx] = eigenvector
            self.projected_data = np.array(X.dot(self.projection_matrix))

        # Reconstruct the required images
        for im in range((self.projected_data.shape[0])):
            projected_image = np.expand_dims(self.projected_data[im], 0)  # (1, num_dims)

            # Matrix multiply projected_image(1, num_dims) with projection_matrix transposed
            reconstructed_image = projected_image.dot(self.projection_matrix.T)
            reconstructed_image = reconstructed_image.reshape(self.X_og_dim[1], self.X_og_dim[2])
            self.compressed_data.append(reconstructed_image)

        return self.projected_data, self.compressed_data


#  # --------- For DEBUGGING -----------#
# a = PCA()
# a.__int__()
# X_pca = a.transform(a.X)
#
#
# print("PRINCIPAL COMPONENT ANALYSIS")
# print('Total Dimensions: {}'.format(len(a.eigen_pair)))
# print('Required Dimensions: {}'.format(a.required_dim))
# print('Training data shape before PCA: {}'.format(np.shape(a.X)))
# print('Training data shape after PCA: {}'.format(a.projected_data.shape))
#
#
# index = 399
# cv.imshow('PCA Dimension Reduced Reconstructed', X_pca[index])
# cv.imshow('train', train_data['train'][index][0])
# cv.waitKey(0)
# print(np.shape(a.X[:][1]))
# test = np.array(a.X[1][:])
# test = test.reshape((24, 21))
# test2 = np.array(a.projected_data)
# # test2 = test2.reshape((18, int(a.required_dim/18)))
# # cv.imshow('test', test)
# # cv.imshow('tert2', test2)
# # cv.waitKey(0)
# print(np.shape(test2))
# print(np.shape(train_data['train'][0][0]))
