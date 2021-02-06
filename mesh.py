import numpy as np
from scipy.spatial import Delaunay


class IsotropicMesh:
    def __init__(self, nodes, infimum=1e-8):
        """
        +--------------+------------------+-------+
        | Tensor       | shape            | type  |
        +--------------+------------------+-------+
        | nodes        | [NN, ND]         | float |
        | mask         | [NN]             | bool  |
        | simplices    | [NT, ND+1]       | int   |
        | tensor       | [NT, ND+1, ND+1] | float |
        | determinants | [NT]             | float |
        | inverses     | [NT, ND+1, ND+1] | float |
        +--------------+------------------+-------+
        """
        self.nodes, (self.nn, self.dim) = nodes, nodes.shape

        # generate anisotropic simplices
        delaunay = Delaunay(self.nodes)
        simplices = delaunay.simplices
        volumes = np.linalg.det(self.nodes[simplices[:, :-1]] - self.nodes[simplices[:, [-1]]])
        valid_indices = [i for i, v in enumerate(volumes) if np.abs(v) > infimum]
        simplices = simplices[valid_indices, :]
        volumes = volumes[valid_indices]

        # generate mask of convex hull
        mask = np.expand_dims(self.nodes[delaunay.convex_hull], axis=1) - np.expand_dims(self.nodes, axis=(0, 2))
        self.masks = np.min(np.abs(np.linalg.det(mask)), axis=0) == 0

        # generate isotropic simplices
        def reverse(spx): return spx[[0, 2, 1] + list(range(3, spx.__len__()))]
        self.simplices = np.array([spx if flag else reverse(spx) for spx, flag in zip(simplices, volumes > 0)])
        self.nt = self.simplices.shape[0]

        # generate minors and determinants of `tensor` in isotropic mode.
        #                   +- x_{t,0,0}  & \cdots & x_{t,0,ND-1}  & 1      -+
        # tensor[t, :, :] = |  \vdots     & \ddots & \vdots        & \vdots  |
        #                   +- x_{t,ND,0} & \cdots & x_{t,ND,ND-1} & 1      -+
        tensor = np.concatenate([self.nodes[self.simplices, :], np.ones(shape=(self.simplices.__len__(), self.dim + 1, 1))], axis=-1)
        self.determinants = np.abs(volumes)
        self.inverses = np.transpose(np.linalg.inv(tensor), axes=(0, 2, 1))


def unit_test():
    def factorial(k): return 1 if k <= 1 else k * factorial(k - 1)

    def estimate_integer(func, points, num_refine=0):
        # convert to vector function
        is_scalar = func(points).shape.__len__() == 1
        if is_scalar:
            def vec_func(x): return np.reshape(func(x), (-1, 1))
        else:
            vec_func = func

        # refine current simplex
        dim = points.shape[-1]
        while num_refine > 0:
            nn = points.shape[0]
            edges = Delaunay(points).simplices[:, [[i, j] for i in range(1, dim + 1) for j in range(i)]]  # [NT, NE', 2]
            indices = [[i // nn, i % nn] for i in set(np.reshape(edges[:, :, 0] * nn + edges[:, :, 1], (-1,)))]
            points = np.vstack([points, np.mean(points[indices, :], axis=1)])  # [NN, ND]
            num_refine -= 1

        # compute all integers in fine simplices.
        simplices = Delaunay(points).simplices  # [NT, ND+1]
        tensor = vec_func(points[simplices.flatten(), :])  # [NT*(ND+1), NF]
        tensor = np.reshape(tensor, newshape=(simplices.shape[0], simplices.shape[1], -1))  # [NT, ND+1, NF]
        volumes = np.abs(np.linalg.det(points[simplices[:, :-1], :] - points[simplices[:, [-1]], :]))  # [NT]
        tensor = 1 / factorial(dim) * np.einsum("tvd,t->vd", tensor, volumes)

        return np.mean(tensor) if is_scalar else np.mean(tensor, axis=0)

    # check method `estimate_integer`:
    #                                                 \Pi_i \alpha_i!
    # \int_K \Pi_i \lambda_i ^ {\alpha_i}  dxdy = ------------------------ * determinant
    #                                             (dim + \Sum_i \alpha_i)!
    for ix in range(3):
        for jx in range(3):
            if ix + jx > 2:
                continue
            result = estimate_integer(
                func=lambda x: x[:, 0] ** ix * x[:, 1] ** jx,
                points=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]),
                num_refine=2 + ix + jx
            )
            print(ix, jx, result)
            assert np.abs(result - factorial(ix) * factorial(jx) / factorial(3 + ix + jx)) / result < 0.01


if __name__ == "__main__":
    unit_test()