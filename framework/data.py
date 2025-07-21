import numpy as np

def generate_synthetic_data(N, sigma, extra_dims):
    N_all = N
    N = int(N_all / 2)
    theta = np.pi * np.random.rand(N, 1)
    z1 = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
    z2 = np.concatenate((np.cos(theta), -np.sin(theta)), axis=1) + np.asarray([1.0, 0.25]).reshape(-1, 1).T
    z = np.concatenate((z1, z2), axis=0) + sigma * np.random.randn(int(N * 2), 2)
    z = z - z.mean(0).reshape(1, -1)
    z3 = (np.sin(np.pi * z[:, 0])).reshape(-1, 1)
    z3 = z3 + sigma * np.random.randn(z3.shape[0], 1)
    data = np.concatenate((z, 0.5 * z3), axis=1)
    if extra_dims > 0:
        noise = sigma * np.random.randn(N_all, extra_dims)
        data = np.concatenate((data, noise), axis=1)
    labels = np.concatenate((0 * np.ones((z1.shape[0], 1)), np.ones((z2.shape[0], 1))), axis=0)
        
    return data, labels