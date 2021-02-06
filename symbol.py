from functools import reduce
import numpy as np
from sparse import SparseTensor


def reduce_prod(seq): return reduce(lambda item_1, item_2: item_1 * item_2, seq)


class Polynomial:
    def __init__(self, coeff, indices, merge=True):
        """\\sum_{i=0}^{N-1} coeff[i] \\Pi_{j=0}^{NV-1} x_j^{indices[i, j]}"""
        self.degree = np.max(np.sum(indices, axis=-1))
        self.n_elements = indices.shape[-1]

        if merge:
            self.coeff, self.indices = SparseTensor.merge(coeff, indices)
        else:
            self.coeff, self.indices = coeff, indices

    def __call__(self, x):
        coeff = np.reshape(self.coeff, newshape=(1, -1))
        x = np.reshape(x, newshape=(-1, 1, self.n_elements))
        indices = np.reshape(self.indices, newshape=(1, -1, self.n_elements))
        return np.sum(coeff * np.prod(np.power(x, indices), axis=2), axis=1)

    def __str__(self): return '\n'.join(["{:.2f}\t{}".format(c, index) for c, index in zip(self.coeff, self.indices)])

    def __neg__(self): return Polynomial(-self.coeff, self.indices, merge=False)

    def __add__(self, other):
        if type(other).__name__ in ["bool", "int", "float", "int64", "float64"]:
            other = Polynomial(np.array([other, ]), np.zeros(shape=(1, self.n_elements), dtype=self.indices.dtype), merge=False)
            return self.__add__(other)
        elif isinstance(other, Polynomial):
            assert self.n_elements == other.n_elements
            return Polynomial(np.hstack([self.coeff, other.coeff]), np.vstack([self.indices, other.indices]), merge=True)
        else:
            raise ValueError

    def __sub__(self, other): return self.__add__(other.__neg__())

    def __mul__(self, other):
        if type(other).__name__ in ["bool", "int", "float", "int64", "float64"]:
            return Polynomial(self.coeff * other, self.indices, merge=False)
        elif isinstance(other, Polynomial):
            assert self.n_elements == other.n_elements
            coeff = np.expand_dims(self.coeff, axis=0) * np.expand_dims(other.coeff, axis=1)
            coeff = coeff.flatten()

            indices = np.expand_dims(self.indices, axis=0) + np.expand_dims(other.indices, axis=1)
            indices = np.reshape(indices, newshape=(-1, self.n_elements))

            return Polynomial(coeff, indices, merge=True)
        else:
            raise ValueError

    def derivative(self, order=1):
        """
        +----------+-----------------+--------------+
        | item     | data type       | shape        |
        +----------+-----------------+--------------+
        | order    | int             | []           |
        | return   | PolynomialArray | [ND] * order |
        +----------+-----------------+--------------+
        """
        array = [self]
        for _ in range(order):
            collection = []
            for poly in array:
                for i in range(self.indices.shape[1]):
                    coeff = poly.coeff * poly.indices[:, i]
                    indices = np.maximum(poly.indices - np.eye(poly.n_elements, dtype=poly.indices.dtype)[[i], :], 0)
                    collection.append(Polynomial(coeff, indices, merge=True))
            array = collection
        return PolynomialArray(array, shape=[self.indices.shape[1]] * order)

    def directional_derivative(self, c, order=1):
        """
        +----------+---------------+--------------+
        | item     | data type     | shape        |
        +----------+---------------+--------------+
        | order    | numpy.ndarray | [ND] * order |
        | order    | int           | []           |
        | return   | Polynomial    | [ND] * order |
        +----------+---------------+--------------+

        return: \\sum_{ij...} c_{ij...} \\frac{\\partial^ self}{\partial \lambda_i \partial \lambda_j ...}
        """
        coeff = self.coeff
        indices = self.indices
        dim = self.n_elements

        for axis in range(order):
            coeff = np.expand_dims(coeff, axis=0) * np.transpose(indices, axes=[-1] + list(range(axis+1)))
            indices = np.expand_dims(indices, axis=0) - np.expand_dims(np.eye(dim, dtype=np.int), axis=list(range(1, axis + 2)))
            indices = np.maximum(indices, 0)
        coeff = (np.expand_dims(c, axis=-1) * coeff).flatten()
        indices = np.reshape(indices, newshape=(-1, dim))

        return Polynomial(coeff, indices, merge=True)


class PolynomialArray:
    def __init__(self, array, shape): self.array, self.shape = array, list(shape)

    def reshape(self, shape):
        shape = list(shape)
        for axis in range(shape.__len__()):
            if shape[axis] == -1:
                shape[axis] = -reduce_prod(self.shape) // reduce_prod(shape)
                break
        return PolynomialArray(self.array, shape)

    def transpose(self, axes):
        transpose_indices = np.transpose(np.reshape(np.arange(self.array.__len__()), newshape=self.shape), axes=axes)
        array = [self.array[index] for index in transpose_indices.flatten()]
        shape = [self.shape[axis] for axis in axes]
        return PolynomialArray(array, shape)

    def sum(self, axis, keep_dim=False):
        axes = [axis] + [ax for ax in range(self.shape.__len__()) if ax != axis]
        transpose_array = self.transpose(axes)

        result = reduce(lambda u, v: u + v, [transpose_array[k] for k in range(transpose_array.shape[0])])
        if keep_dim:
            result.shape.insert(axis, 1)
        return result

    def __call__(self, x): return np.reshape(np.stack([poly(x) for poly in self.array], axis=1), newshape=[-1] + self.shape)

    def __getitem__(self, item):
        valid_indices = np.reshape(np.arange(self.array.__len__()), newshape=self.shape)[item]
        array = [self.array[index] for index in valid_indices.flatten()]
        shape = valid_indices.shape
        return array[0] if shape == () else PolynomialArray(array, shape)

    def __eq__(self, other): return (self.shape == other.shape) and sum([sp != op for sp, op in zip(self.array, other.array)]) == 0

    def __neg__(self): return PolynomialArray([-array for array in self.array], self.shape)

    def __add__(self, other):  # TODO: in large scale calculation, this operator works slowly in serial mode.
        if type(other).__name__ in ["bool", "int", "float", "Polynomial"]:
            array = PolynomialArray([sa + other for sa in self.array], self.shape)
            return array.reshape(self.shape)
        elif isinstance(other, np.ndarray):
            n_elements, dtype = self.array[0].n_elements, self.array[0].indices.dtype
            arr = [Polynomial(np.array([item, ]), np.zeros(shape=(1, n_elements), dtype=dtype)) for item in other.flatten()]
            return self.__add__(PolynomialArray(arr, shape=other.shape))
        elif isinstance(other, PolynomialArray):
            self_indices = np.reshape(np.arange(np.prod(self.shape)), self.shape)
            o_indices = np.reshape(np.arange(np.prod(other.shape)), other.shape)
            self_indices, o_indices = self_indices + np.zeros_like(o_indices), o_indices + np.zeros_like(self_indices)

            array = [self.array[si] + other.array[oi] for si, oi in zip(self_indices.flatten(), o_indices.flatten())]
            return PolynomialArray(array, shape=self_indices.shape)
        else:
            raise ValueError

    def __sub__(self, other): return self.__add__(other.__neg__())

    def __mul__(self, other):  # TODO: in large scale calculation, this operator works slowly in serial mode.
        if type(other).__name__ in ["bool", "int", "float", "Polynomial"]:
            array = PolynomialArray([sa * other for sa in self.array], self.shape)
            return array.reshape(self.shape)
        elif isinstance(other, np.ndarray):
            n_elements, dtype = self.array[0].n_elements, self.array[0].indices.dtype
            arr = [Polynomial(np.array([item, ]), np.zeros(shape=(1, n_elements), dtype=dtype)) for item in other.flatten()]
            return self.__mul__(PolynomialArray(arr, shape=other.shape))
        elif isinstance(other, PolynomialArray):
            self_indices = np.reshape(np.arange(np.prod(self.shape)), self.shape)
            o_indices = np.reshape(np.arange(np.prod(other.shape)), other.shape)
            self_indices, o_indices = self_indices + np.zeros_like(o_indices), o_indices + np.zeros_like(self_indices)

            array = [self.array[si] * other.array[oi] for si, oi in zip(self_indices.flatten(), o_indices.flatten())]
            return PolynomialArray(array, shape=self_indices.shape)
        else:
            raise ValueError

    @classmethod
    def stack(cls, arrays, axis):
        axis %= arrays[0].shape.__len__() + 1
        array = sum([item.array for item in arrays], [])
        shape = [arrays.__len__()] + list(arrays[0].shape)
        axes = [i for i in range(shape.__len__()) if i != axis]
        axes.insert(axis, 0)
        return PolynomialArray(array, shape).transpose(axes)

    @classmethod
    def concat(cls, arrays, axis):
        axes = [axis] + [i for i in range(arrays[0].shape.__len__()) if i != axis]
        shape = [-1] + [dim for i, dim in enumerate(arrays[0].shape) if i != axis]
        arrays = sum([cls.transpose(array, axes).array for array in arrays], [])
        arrays = cls(arrays, shape=(arrays.__len__(), ))
        arrays = arrays.reshape(shape)

        axes = list(range(1, shape.__len__()))
        axes.insert(axis, 0)
        return arrays.transpose(axes)

    def derivative(self, order=1):
        """
        +----------+-----------------+---------------------------+
        | item     | data type       | shape                     |
        +----------+-----------------+---------------------------+
        | order    | int             | []                        |
        | return   | PolynomialArray | self.shape + [ND] * order |
        +----------+-----------------+---------------------------+
        """
        array = PolynomialArray.stack([poly.derivative(order) for poly in self.array], axis=0)
        return array.reshape(self.shape + array.shape[1:])

    def directional_derivative(self, c, order=1):
        """
        +----------+-----------------+---------------------------+
        | item     | data type       | shape                     |
        +----------+-----------------+---------------------------+
        | c        | numpy.ndarray   | self.shape + [ND] * order |
        | order    | int             | []                        |
        | return   | numpy.ndarray   | self.shape                |
        +----------+-----------------+---------------------------+

        return: \\sum_{ij...} c_{ij...}^{uv...} \\frac{\\partial^ self_{uv...}}{\partial \lambda_i \partial \lambda_j ...}
        """
        ni = max([p.coeff.__len__() for p in self.array])
        dim = self.array[0].n_elements
        coeff = [np.concatenate([p.coeff, np.zeros(shape=(ni - p.coeff.__len__(), ))], axis=0) for p in self.array]
        coeff = np.stack(coeff, axis=1)  # shape = [NI, ?]
        indices = [np.concatenate([p.indices, np.zeros(shape=(ni - p.coeff.__len__(), dim), dtype=np.int)], axis=0) for p in self.array]
        indices = np.stack(indices, axis=2)  # shape = [NI, ND, ?]

        for axis in range(order):
            axes = [axis + 1] + [i for i in range(axis + 3) if i != axis + 1]
            coeff = np.expand_dims(coeff, axis=0) * np.transpose(indices, axes=axes)
            axes = list(range(1, axis + 2)) + [axis + 3]
            indices = np.expand_dims(indices, axis=0) - np.expand_dims(np.eye(dim, dtype=np.int), axis=axes)
            indices = np.maximum(indices, 0)

        c = np.reshape(c, newshape=[-1, 1] + [dim] * order)
        c = np.transpose(c, axes=list(range(2, order + 2)) + [1, 0])  # shape = [ND] * order + [1] + [?]
        coeff = np.reshape((c * coeff), newshape=(dim ** order * ni, -1))  # shape = [ND] * order + [NI] + [?]
        indices = np.reshape(indices, newshape=(dim ** order * ni, dim, -1))  # shape = [ND] * order + [NI] + [ND] + [?]

        return PolynomialArray([Polynomial(coeff[:, i], indices[:, :, i], merge=True) for i in range(coeff.shape[-1])], shape=self.shape)

    def integral(self, dim, determinant):
        """
        Working correctly in triangulation grid only!
                                                   \Pi_i \alpha_i!
        \int_K \Pi_i \lambda_i^{\alpha_i} dx = ------------------------ * determinant
                                               (dim + \Sum_i \alpha_i)!
        """
        ni = max([p.coeff.__len__() for p in self.array])
        nd = self.array[0].n_elements
        coeff = [np.concatenate([p.coeff, np.zeros(shape=(ni - p.coeff.__len__(), ))], axis=0) for p in self.array]
        coeff = np.stack(coeff, axis=1)  # shape = [NI, ?]
        indices = [np.concatenate([p.indices, np.zeros(shape=(ni - p.coeff.__len__(), nd), dtype=np.int)], axis=0) for p in self.array]
        indices = np.stack(indices, axis=2)  # shape = [NI, ND, ?]
        degree = np.max(indices)

        if degree == 0:
            numerator = np.ones_like(indices)  # shape = [NI, ND, ?]
        else:
            numerator = reduce_prod([np.maximum(indices - i, 1) for i in range(degree)])  # shape = [NI, ND, ?]
        numerator = np.prod(numerator, axis=1)  # shape = [NI, ?]

        denominator = np.sum(indices, axis=1) + dim  # shape = [NI, ?]
        denominator = reduce_prod([np.maximum(denominator - i, 1) for i in range(degree + dim)])  # shape = [NI, ?]

        return np.reshape(np.sum(coeff * numerator / denominator, axis=0), newshape=self.shape) * determinant


def unit_test():
    np.set_printoptions(precision=2)
    x = np.random.rand(4, 3)
    const_array = np.random.rand(8, 7)

    # item 6, degree 2, elements 3
    poly = Polynomial(coeff=np.random.rand(6), indices=np.random.randint(0, 3, size=(6, 3)))

    polys_1 = [Polynomial(coeff=np.random.rand(5), indices=np.random.randint(0, 5, size=(5, 3))) for _ in range(56)]
    polys_1 = PolynomialArray(polys_1, [8, 7])

    polys_2 = [Polynomial(coeff=np.random.rand(4), indices=np.random.randint(0, 5, size=(4, 3))) for i in range(56)]
    polys_2 = PolynomialArray(polys_2, [8, 7])

    polys_3 = [Polynomial(coeff=np.random.rand(3), indices=np.random.randint(0, 5, size=(3, 3))) for i in range(7*8*9)]
    polys_3 = PolynomialArray(polys_3, [9, 8, 7])

    # four fundamental rules
    print("polys_1(x) + np.pi - (polys_1 + np.pi)(x):")
    print(np.max(np.abs(polys_1(x) + np.pi - (polys_1 + np.pi)(x))))
    print("polys_1(x) + poly(x) - (polys_1 + poly)(x):")
    print(np.max(np.abs(polys_1(x) + np.reshape(poly(x), (-1, 1, 1)) - (polys_1 + poly)(x))))
    print("polys_1(x) + np.expand_dims(const_array, axis=0) - (polys_1 + const_array)(x):")
    print(np.max(np.abs(polys_1(x) + np.expand_dims(const_array, axis=0) - (polys_1 + const_array)(x))))
    print("polys_1(x) + polys_2(x) - (polys_1 + polys_2)(x):")
    print(np.max(np.abs(polys_1(x) + polys_2(x) - (polys_1 + polys_2)(x))))

    print("polys_1[:, [1]](x) + polys_2[[-1], :](x) - (polys_1[:, [1]] + polys_2[[-1], :])(x):")
    print(np.max(np.abs(polys_1[:, [1]](x) + polys_2[[-1], :](x) - (polys_1[:, [1]] + polys_2[[-1], :])(x))))

    print("polys_1(x) - np.pi - (polys_1 - np.pi)(x):")
    print(np.max(np.abs(polys_1(x) - np.pi - (polys_1 - np.pi)(x))))
    print("polys_1(x) - poly(x) - (polys_1 - poly)(x):")
    print(np.max(np.abs(polys_1(x) - np.reshape(poly(x), (-1, 1, 1)) - (polys_1 - poly)(x))))
    print("polys_1(x) - np.expand_dims(const_array, axis=0) - (polys_1 - const_array)(x):")
    print(np.max(np.abs(polys_1(x) - np.expand_dims(const_array, axis=0) - (polys_1 - const_array)(x))))
    print("polys_1(x) - polys_2(x) - (polys_1 - polys_2)(x):")
    print(np.max(np.abs(polys_1(x) - polys_2(x) - (polys_1 - polys_2)(x))))

    print("polys_1[:, [1]](x) - polys_2[[-1], :](x) - (polys_1[:, [1]] - polys_2[[-1], :])(x):")
    print(np.max(np.abs(polys_1[:, [1]](x) - polys_2[[-1], :](x) - (polys_1[:, [1]] - polys_2[[-1], :])(x))))

    print("polys_1(x) * np.pi - (polys_1 * np.pi)(x):")
    print(np.max(np.abs(polys_1(x) * np.pi - (polys_1 * np.pi)(x))))
    print("polys_1(x) * poly(x) - (polys_1 * poly)(x):")
    print(np.max(np.abs(polys_1(x) * np.reshape(poly(x), (-1, 1, 1)) - (polys_1 * poly)(x))))
    print("polys_1(x) * np.expand_dims(const_array, axis=0) - (polys_1 * const_array)(x):")
    print(np.max(np.abs(polys_1(x) * np.expand_dims(const_array, axis=0) - (polys_1 * const_array)(x))))
    print("polys_1(x) * polys_2(x) - (polys_1 * polys_2)(x):")
    print(np.max(np.abs(polys_1(x) * polys_2(x) - (polys_1 * polys_2)(x))))

    print("polys_1[:, [1]](x) * polys_2[[-1], :](x) - (polys_1[:, [1]] * polys_2[[-1], :])(x):")
    print(np.max(np.abs(polys_1[:, [1]](x) * polys_2[[-1], :](x) - (polys_1[:, [1]] * polys_2[[-1], :])(x))))

    print(np.max(np.abs(polys_1.reshape(shape=[2, 4, 7])(x) - np.reshape(polys_1(x), newshape=(-1, 2, 4, 7)))))

    # check concat
    print("PolynomialArray.concat([polys_1, polys_2], axis=1)(x) - np.concatenate([polys_1(x), polys_2(x)], axis=1):")
    print(np.max(np.abs(PolynomialArray.concat([polys_1, polys_2], axis=1)(x) - np.concatenate([polys_1(x), polys_2(x)], axis=2))))

    # check sum
    print(np.max(np.abs(polys_3.sum(axis=0, keep_dim=True)(x) - np.sum(polys_3(x), axis=0 + 1, keepdims=True))))
    print(np.max(np.abs(polys_3.sum(axis=1, keep_dim=True)(x) - np.sum(polys_3(x), axis=1 + 1, keepdims=True))))
    print(np.max(np.abs(polys_3.sum(axis=2, keep_dim=True)(x) - np.sum(polys_3(x), axis=2 + 1, keepdims=True))))

    # check integral
    poly_1 = Polynomial(
        coeff=np.array([
            1,
            3,
        ]),
        indices=np.array([
            [1, 2, 3, 4],
            [1, 1, 1, 1],
        ])
    )
    poly_2 = Polynomial(
        coeff=np.array([
            2,
            4,
        ]),
        indices=np.array([
            [4, 3, 2, 1],
            [0, 0, 0, 0],
        ])
    )
    poly = PolynomialArray(array=[poly_1, poly_2], shape=(2, ))
    ans_1 = 0.5 * 1 * (1 * 2 * 6 * 24) / reduce_prod(list(range(1, 14)))
    ans_1 += 0.5 * 3 * (1 * 1 * 1 * 1) / reduce_prod(list(range(1, 8)))
    ans_2 = 2 * 2 * (1 * 2 * 6 * 24) / reduce_prod(list(range(1, 14)))
    ans_2 += 2 * 4 * (1 * 1 * 1 * 1) / reduce_prod(list(range(1, 4)))
    print(poly.integral(dim=3, determinant=np.array([0.5, 2])) - np.array([ans_1, ans_2]))

    # check derivative
    poly = poly.derivative(order=1)
    print(poly[0, 1])

    # check derivative in Polynomial
    c = np.random.rand(3, 3)
    coeff = np.random.randint(100, size=(4, )) / 100
    indices = np.random.randint(10, size=(4, 3))

    poly = Polynomial(coeff, indices)
    type_1 = (poly.derivative(order=2) * c).sum(axis=0).sum(axis=0)
    type_2 = poly.directional_derivative(c, order=2)
    error = type_1 - type_2
    error = Polynomial(error.coeff, error.indices, merge=True)
    print("error:", error)

    # check derivative in PolynomialArray
    poly = PolynomialArray([poly, poly+1, poly-1, poly*2], shape=(2, 2))
    c = np.random.rand(2, 2, 3, 3)
    type_1 = (poly.derivative(order=2) * c).sum(axis=2).sum(axis=2)
    type_2 = poly.directional_derivative(c, order=2)
    for item in (type_1 - type_2).array:
        item = Polynomial(item.coeff, item.indices, merge=True)
        print("error:", item)


if __name__ == "__main__":
    unit_test()
