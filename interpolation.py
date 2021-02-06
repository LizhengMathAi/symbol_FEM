import numpy as np
from sparse import SparseTensor
from symbol import reduce_prod, Polynomial, PolynomialArray
from mesh import IsotropicMesh


class BasisFunctions:
    @classmethod
    def integer2base(cls, n, base, width):
        if not isinstance(n, list):
            n = [n]
        if n.__len__() != width + 1:
            return cls.integer2base(n=[n[-1] % base] + n[:-1] + [n[-1] // base], base=base, width=width)
        else:
            return np.stack(n[:-1], axis=-1)

    @classmethod
    def base2integer(cls, n, base):
        shape = n.shape
        n = np.reshape(n, newshape=(-1, shape[-1]))
        integer = n[:, 0]
        for i in range(1, shape[-1]):
            integer = integer * base + n[:, i]
        return np.reshape(integer, newshape=shape[:-1])

    def __init__(self, degree: int, mesh: IsotropicMesh, broadcast=True):
        """
        * If the shape function have form that not relevant to simplex, then `broadcast` == True and
        self.shape_functions.shape == [NF]. If not, `broadcast` == False and self.shape_functions.shape == [NT, NF].

        For instance, in Lagrange interpolation, the `broadcast` is True. But in Hermite interpolation it's False.

        +-----------------+----------------------+--------------------+---------------------------------------------------------------+
        | item            | type                 | shape              | annotation                                                    |
        +-----------------+----------------------+--------------------+---------------------------------------------------------------+
        | simplex_dof     | numpy.ndarray(int)   | [NF, ]             | NF = degree of freedoms in each simplex                       |
        | dof_points      | numpy.ndarray(float) | [NF, ND + 1]       | weights, the coefficients of interpolation points             |
        | dof_orders      | numpy.ndarray(int)   | [NF, ]             | orders in each interpolation points in any simplex            |
        | shape_functions | PolynomialArray      | [NF, ] or [NT, NF] |  shape functions in each simplex                              |
        | map             | dict                 | (t, f) -> f'       | (simplex_index, shape_function_index) -> basis_function_index |
        | domain_dof      | numpy.ndarray(int)   | [NF', ]            | NF' = degree of freedoms in global domain                     |
        +-----------------+----------------------+--------------------+---------------------------------------------------------------+
        """
        self.mesh = mesh
        self.broadcast = broadcast

        self.simplex_dof, self.dof_points, self.dof_orders = None, None, None
        self.shape_functions = None  # shape = [NT, ] or shape = [NT, NF]
        self.map, self.boundary_indices, self.inner_indices, self.domain_dof = None, None, None, None

    def gen_graph(self):
        """
        sparse_map[simplex_index, point_index in simplex] = element_index in domain
        mask_list = [(simplex_index, point_index in simplex) is boundary node, ...]
        """
        sparse_map, key_pairs, mask_list = np.zeros(shape=(self.mesh.nt, self.simplex_dof), dtype=np.int), [], []
        for i, simplex in enumerate(self.mesh.simplices):
            for j, weights in enumerate(self.dof_points):
                node_key = np.where(weights != 0, simplex, -1)
                arg_sort = np.argsort(node_key)
                node_key = node_key[arg_sort]
                anchor_key = weights[arg_sort]
                pair = str(list(node_key) + list(anchor_key)) + str(self.dof_orders[j])
                try:
                    sparse_map[i, j] = key_pairs.index(pair)
                except ValueError:
                    sparse_map[i, j] = key_pairs.__len__()
                    key_pairs.append(pair)
                    mask_list.append(sum([self.mesh.masks[i] for i in node_key if i != -1]) == sum([i != -1 for i in node_key]))
        boundary_indices = [i for i, mask in enumerate(mask_list) if mask]
        inner_indices = [i for i, mask in enumerate(mask_list) if not mask]
        domain_dof = key_pairs.__len__()
        return sparse_map, boundary_indices, inner_indices, domain_dof

    def linear_combination(self, c, order):
        """
        +----------+-----------------+------------------+
        | item     | data type       | shape            |
        +----------+-----------------+------------------+
        | c        | numpy.ndarray   | [ND] * order     |
        | order    | int             | []               |
        | return   | numpy.ndarray   | [NT, NF]         |
        +----------+-----------------+------------------+

        c * d_xyz = einsum('pABC,tAa,tBb,tCc,abc->tp', d_lambda, jacobi, jacobi, jacobi, c)  # order = 3
                  = einsum('tABC,pABC->tp', einsum('abc,tAa,tBb,tCc->tABC', c, jacobi, jacobi, jacobi), d_lambda)
        """
        jacobi = self.mesh.inverses[:, :, :-1]  # shape = [NT, ND + 1, ND]

        if order == 0:
            if self.broadcast:
                return self.shape_functions.reshape([1, -1]) * np.ones(shape=(self.mesh.nt, 1)) * c
            else:
                return self.shape_functions * c

        command = ''.join([chr(97 + i) for i in range(order)]) + ',' + ','.join(
            ['z' + chr(65 + i) + chr(97 + i) for i in range(order)]) + '->z' + ''.join(
            [chr(65 + i) for i in range(order)])
        c = np.einsum(command, c, *([jacobi] * order))  # shape: [NT, ] + [ND, ] * order
        c = np.stack([c] * self.simplex_dof, axis=1)  # shape: [NT, NF] + [ND, ] * order
        if self.broadcast:
            shape_functions = PolynomialArray.stack([self.shape_functions] * self.mesh.nt, axis=0)  # shape = [NT, NF]
            return shape_functions.directional_derivative(c=c, order=order)  # shape = [NT, NF]
        else:
            return self.shape_functions.directional_derivative(c=c, order=order)  # shape = [NT, NF]

    @classmethod
    def dense_integral(cls, combinations, mesh):
        """
        +----------+-----------------+---------------------------------+
        | item     | data type       | shape                           |
        +----------+-----------------+---------------------------------+
        | return   | numpy.ndarray   | [NT, NF_0, NF_1, ..., NF_{k-1}] |  # k = the length of each sequence.
        +----------+-----------------+---------------------------------+

        return: \\int einsum('ti,tj,tk,...->tijk...', c_i * d^{i}_xyz, c_j * d^{j}_xyz, c_k * d^{k}_xyz, ...)
        """
        tensor = 1
        for i, comb in enumerate(combinations):
            nt, dof = comb.shape
            shape = [nt] + [1] * combinations.__len__()
            shape[i + 1] = dof
            tensor = comb.reshape(shape) * tensor
        determinants = np.reshape(mesh.determinants, newshape=[-1, ] + [1] * combinations.__len__())

        return tensor.integral(dim=mesh.dim, determinant=determinants)

    @classmethod
    def sparse_integral(cls, dense_integral, shape_functions_seq):
        """
        +----------------+---------------+---------------------------------+
        | item           | data type     | shape                           |
        +----------------+---------------+---------------------------------+
        | dense_integral | numpy.ndarray | [NT, NF_0, NF_1, ..., NF_{k-1}] |  # k = the length of each sequence.
        | return         | SparseTensor  | [NF'_0, NF'_1, ..., NF'_{k-1}]  |  # NF' = the DOFs in global domain.
        +----------------+---------------+---------------------------------+
        """
        dense_shape = dense_integral.shape

        data = np.transpose(dense_integral, axes=list(range(1, dense_shape.__len__())) + [0]).flatten()
        flatten_ids = np.arange(reduce_prod(dense_shape[1:]))
        flatten_ids = SparseTensor.integer2base(flatten_ids, base=dense_shape[1:])
        indices = []
        for dense_ids in flatten_ids:
            indices.append(np.stack([shape_functions_seq[axis].map[:, i] for axis, i in enumerate(dense_ids)], axis=1))
        indices = np.concatenate(indices, axis=0)
        sparse_shape = [e.domain_dof for e in shape_functions_seq]
        return SparseTensor(np.array(data), np.array(indices), sparse_shape)

    def dense_numerical_integral(self, combination, func, mesh):
        """
        +----------------------+---------------+--------------+
        | item                 | data type     | shape        |
        +----------------------+---------------+--------------+
        | combination          | numpy.ndarray | [NT, NF]     |
        | interpolation_points | numpy.ndarray | [NT, NF, ND] |
        | f_val                | numpy.ndarray | [NT, NF]     |
        | return               | numpy.ndarray | [NT, NF]     |
        +----------------------+---------------+--------------+

        return: \\int (c * d_xyz, f)  # `func` must be a scalar function.
        """
        interpolation_points = np.einsum("tvd,fv->tfd", mesh.nodes[mesh.simplices, :], self.dof_points)
        f_val = np.reshape(func(np.reshape(interpolation_points, newshape=(-1, mesh.dim))), newshape=(mesh.nt, -1))
        tensor = combination * f_val
        determinant = np.expand_dims(mesh.determinants, axis=1)

        return tensor.integral(dim=mesh.dim, determinant=determinant)

    @classmethod
    def sparse_numerical_integral(cls, dense_numerical_integral, shape_functions):
        """
        +--------------------------+-----------------+----------+
        | item                     | data type       | shape    |
        +--------------------------+-----------------+----------+
        | dense_numerical_integral | numpy.ndarray   | [NT, NF] |
        | return                   | numpy.ndarray   | [NF', ]  |  # NF' = the DOFs in global domain.
        +--------------------------+-----------------+----------+
        """
        nt, nf = dense_numerical_integral.shape
        sparse_numerical_integral = np.zeros(shape=[shape_functions.domain_dof, ], dtype=dense_numerical_integral.dtype)

        for t in range(nt):
            for i in range(nf):
                sparse_numerical_integral[shape_functions.map[t, i]] += dense_numerical_integral[t, i]

        return sparse_numerical_integral

    def exact2numerical(self, func_u_seq):
        """
        +----------+-----------------+--------------------+
        | returns  | data type       | shape              |
        +----------+-----------------+--------------------+
        | u0_array | numpy.ndarray   | [NT, NF_0]         |  # coefficients in order 0
        | u1_array | numpy.ndarray   | [NT, NF_1, ND]     |  # coefficients in order 1
        | u2_array | numpy.ndarray   | [NT, NF_2, ND, ND] |  # coefficients in order 2
        | ...      | ...             | ...                |
        +----------+-----------------+--------------------+
        """
        u_seq, nt = [], self.mesh.nt
        for k, func_u in enumerate(func_u_seq):
            uk_indices = [i for i, order in enumerate(self.dof_orders) if order == k]
            uk_indices = uk_indices[:uk_indices.__len__() // self.mesh.dim ** k]
            dof_points = np.einsum("tvd,fv->tfd", self.mesh.nodes[self.mesh.simplices], self.dof_points[uk_indices])
            dof_points = np.reshape(dof_points, newshape=(-1, self.mesh.dim))
            u_seq.append(np.reshape(func_u_seq[k](dof_points), newshape=[nt, -1] + [self.mesh.dim] * k))
        return u_seq

    def interpolation(self, weights, u_seq):
        """
        +----------------------+-----------------+--------------------+
        | item                 | data type       | shape              |
        +----------------------+-----------------+--------------------+
        | weights              | numpy.ndarray   | [NW, ND + 1]       |  # weights of new interpolation points in each simplex
        | u0_array             | numpy.ndarray   | [NT, NF_0]         |  # coefficients in order 0
        | u1_array             | numpy.ndarray   | [NT, NF_1, ND]     |  # coefficients in order 1
        | u2_array             | numpy.ndarray   | [NT, NF_2, ND, ND] |  # coefficients in order 2
        | ...                  | ...             | ...                |
        | interpolation_points | numpy.ndarray   | [NT * NW', ND]     |  # new interpolation points in Cartesian coordinates
        | numerical_u_array    | PolynomialArray | [NT]               |
        | numerical_u          | numpy.ndarray   | [NT * NW']         |  # numerical value
        +----------------------+-----------------+--------------------+
        """
        interpolation_points = np.einsum("tvd,wv->twd", self.mesh.nodes[self.mesh.simplices], weights)
        interpolation_points = np.reshape(interpolation_points, newshape=[-1, self.mesh.dim])

        nt = u_seq[0].shape[0]
        coeff = np.concatenate([np.reshape(uk_array, newshape=[nt, -1]) for uk_array in u_seq], axis=1)
        if self.broadcast:
            numerical_u_array = (self.shape_functions.reshape([1, -1]) * coeff).sum(axis=1)
        else:
            numerical_u_array = (self.shape_functions * coeff).sum(axis=1)
        numerical_u = np.transpose(numerical_u_array(weights), axes=[1, 0]).flatten()

        return interpolation_points, numerical_u

    def error(self, coeff, func_u, order):
        """
        +---------------+-----------------+-------------+
        | item          | data type       | shape       |
        +---------------+-----------------+-------------+
        | coeff         | numpy.ndarray   | [NF']       |  # numerical coefficients of domain interpolation points
        | u0_array      | numpy.ndarray   | [NT, NF]    |  # numerical coefficients of simplex interpolation points
        | domain_points | numpy.ndarray   | [NT*NF, ND] |  # interpolation points in Cartesian coordinates
        | exact_u       | numpy.ndarray   | [NT*NF]     |  # exact value
        | numerical_u   | numpy.ndarray   | [NT*NF]     |  # numerical value
        | error         | float           | []          |  # error estimate in L_{order} norm
        +---------------+-----------------+-------------+
        """
        u0_array = np.zeros([self.mesh.nt, self.simplex_dof], dtype=np.float)
        for t in range(self.mesh.nt):
            for f in range(self.simplex_dof):
                u0_array[t, f] = coeff[self.map[t, f]]

        exact_u0_array, = self.exact2numerical([func_u, ])
        domain_points, exact_u = self.interpolation(weights=self.dof_points, u_seq=[exact_u0_array])
        domain_points, numerical_u = self.interpolation(weights=self.dof_points, u_seq=[u0_array])

        gap = np.reshape(exact_u - numerical_u, newshape=[self.mesh.nt, self.simplex_dof])
        areas = 1 / reduce_prod(list(range(1, self.mesh.dim + 1))) * self.mesh.determinants
        error = np.sum(np.power(np.mean(np.power(gap, order), axis=1), 1 / order) * areas)

        return domain_points, exact_u, numerical_u, error


class LagrangeBasisFunctions(BasisFunctions):
    def __init__(self, degree, mesh):
        super().__init__(degree, mesh, broadcast=True)

        indices = self.integer2base(np.arange((degree + 1) ** (self.mesh.dim + 1)), base=degree + 1, width=self.mesh.dim + 1)
        indices = indices[np.sum(indices, axis=1) == degree, :]
        self.simplex_dof = indices.__len__()

        polys = PolynomialArray([Polynomial(np.array([1.]), np.expand_dims(index, axis=0)) for index in indices], shape=(self.simplex_dof, ))

        self.dof_points = 1 / degree * indices  # shape = [dof, dimension + 1]
        self.dof_orders = [0] * self.simplex_dof  # shape = [dof, ]
        self.shape_functions = PolynomialArray([Polynomial(c, indices, merge=True) for c in np.linalg.inv(polys(self.dof_points)).T], shape=(self.simplex_dof, ))
        self.map, self.boundary_indices, self.inner_indices, self.domain_dof = self.gen_graph()


class HermiteBasisFunctions(BasisFunctions):
    def __init__(self, degree, mesh):
        assert degree == 3
        super().__init__(degree, mesh, broadcast=False)

        indices = self.integer2base(np.arange((degree + 1) ** (self.mesh.dim + 1)), base=degree + 1, width=self.mesh.dim + 1)
        indices = indices[np.sum(indices, axis=1) == degree, :]  # C_{dim+1}^1 + 2 * C_{dim+1}^2 + C_{dim+1}^3
        self.simplex_dof = indices.__len__()

        # all polynomials
        polys = PolynomialArray(
            [Polynomial(np.array([1.]), np.expand_dims(index, axis=0)) for index in indices],
            shape=(self.simplex_dof, ))

        # all anchors of DOFs
        center_points = self.integer2base(np.arange(2 ** (self.mesh.dim + 1)), base=2, width=self.mesh.dim + 1)
        center_points = 1 / 3 * center_points[np.sum(center_points, axis=1) == degree, :]
        self.dof_points = np.vstack([np.eye(self.mesh.dim + 1), center_points] + [np.eye(self.mesh.dim + 1)] * self.mesh.dim)  # [dof, dimension + 1]
        self.dof_orders = [0] * (self.simplex_dof - (self.mesh.dim + 1) * self.mesh.dim) + [1] * ((self.mesh.dim + 1) * self.mesh.dim)  # shape = [dof, ]

        # shape functions
        u0s = polys(np.vstack([np.eye(self.mesh.dim + 1), center_points]))  # shape = [G0, dof]
        u0s = np.einsum("t,gf->tgf", np.ones((self.mesh.nt,)), u0s)  # shape = [NT, G0, dof]
        u1s = polys.derivative(order=1)(np.eye(self.mesh.dim + 1))  # shape = [G1, dof, vertices]
        u1s = np.einsum("pfv,tvn->tpnf", u1s, self.mesh.inverses[:, :, :-1])  # shape = [NT, G1, dimension, dof]
        u1s = np.reshape(u1s, newshape=(self.mesh.nt, (self.mesh.dim + 1) * self.mesh.dim, -1))  # shape = [NT, G1 * dimension, dof]

        tensor = np.concatenate([u0s, u1s], axis=1)  # shape = [NT, dof, dof]
        coeff = np.transpose(np.linalg.inv(tensor), axes=(0, 2, 1))  # shape = [NT, dof, dof]

        self.shape_functions = PolynomialArray([Polynomial(c, indices, merge=True) for cs in coeff for c in cs], shape=[self.mesh.nt, self.simplex_dof])

        self.map, self.boundary_indices, self.inner_indices, self.domain_dof = self.gen_graph()


def unit_test():
    from matplotlib import pyplot as plt
    from demo import cassini_oval_region

    np.set_printoptions(precision=2, linewidth=240)

    mesh = IsotropicMesh(nodes=cassini_oval_region(num=16))

    def func_u(x): return np.exp(-np.sum(np.square(x), axis=1)) * np.sum(np.cos(x), axis=1)

    def func_u1(x):
        ux = -np.exp(-np.sum(np.square(x), axis=1)) * (2 * x[:, 0] * np.sum(np.cos(x), axis=1) + -np.sin(x[:, 0]))
        uy = -np.exp(-np.sum(np.square(x), axis=1)) * (2 * x[:, 1] * np.sum(np.cos(x), axis=1) + -np.sin(x[:, 1]))
        return np.stack([ux, uy], axis=-1)

    weights = np.array([
        [2 / 3, 1 / 6, 1 / 6],
        [1 / 6, 2 / 3, 1 / 6],
        [1 / 6, 1 / 6, 2 / 3],
    ])

    fig = plt.figure(figsize=(12, 12))

    poly = HermiteBasisFunctions(degree=3, mesh=mesh)
    u0_array, u1_array = poly.exact2numerical(func_u_seq=[func_u, func_u1])
    interpolation_points, numerical_val = poly.interpolation(weights, [u0_array, u1_array])
    exact_val = func_u(interpolation_points)
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.set_title("Hermite error(polynomial degree={}, dof={})".format(3, poly.simplex_dof), fontsize=8)
    ax.plot_trisurf(interpolation_points[:, 0], interpolation_points[:, 1], exact_val - numerical_val, alpha=0.5)

    for degree in [1, 2, 3, 4, 5]:
        poly = LagrangeBasisFunctions(degree=degree, mesh=mesh)
        u0_array, = poly.exact2numerical(func_u_seq=[func_u, ])

        interpolation_points, numerical_val = poly.interpolation(weights, [u0_array])
        ax = fig.add_subplot(2, 3, degree + 1, projection='3d')
        ax.set_title("Lagrange error(polynomial degree={}, dof={})".format(degree, poly.simplex_dof), fontsize=6)
        ax.plot_trisurf(interpolation_points[:, 0], interpolation_points[:, 1], exact_val - numerical_val, alpha=0.5)

    plt.rc("font", size=6)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

    fig.savefig('./interpolation.png')
    plt.show()


if __name__ == "__main__":
    unit_test()
