from functools import reduce
import numpy as np


class SparseTensor:
    """
    In this class:
        * Slice operators are limited.
        * Not allow you to use broadcast operators!
    """

    @classmethod
    def merge(cls, data, indices, eps=1e-12):
        indices, inverse = np.unique(indices, axis=0, return_inverse=True)

        _data = np.zeros(shape=(indices.shape[0], ), dtype=data.dtype)
        for i, v in enumerate(inverse):
            _data[v] += data[i]
        data = _data

        valid_ids = [i for i, c in enumerate(data) if abs(c) > eps]
        if not valid_ids:
            data = np.zeros(shape=(1,), dtype=data.dtype)  # 1D-array
            indices = np.zeros(shape=(1, indices.shape[1]), dtype=indices.dtype)  # 2D-array
        else:
            data = data[valid_ids]
            indices = indices[valid_ids]
        return data, indices

    def __init__(self, data, indices, shape, sparse_merge=True):
        if sparse_merge:
            self.data, self.indices = self.merge(data, indices)
        else:
            self.data, self.indices = data, indices
        self.shape, self.dtype = list(shape), data.dtype

    def __str__(self): return '\n'.join(["{:.2f}\t{}".format(c, index) for c, index in zip(self.data, self.indices)])

    def dense(self):
        dense = np.zeros(shape=self.shape, dtype=self.dtype)
        for data, indices in zip(self.data, self.indices):
            commend = "dense{} += {}".format(list(indices), data)
            exec(commend)
        return dense

    # ------------- slice operation -------------
    def __getitem__(self, item):
        """
        Support operators:
            * SparseTensor[:, [i, j, k], :, [p, q], :, x, ...]
        Not support operators:
            * SparseTensor[:2, 2:, i:j:k, ...]
        """
        if not isinstance(item, tuple):
            item = (item, )

        data, indices, shape = self.data.copy(), self.indices.copy(), self.shape.copy()

        for axis, ids in enumerate(item):
            if isinstance(ids, slice):
                continue
            elif isinstance(ids, list):
                valid_ids = sum([[i for i, index in enumerate(indices[:, axis]) if index == k] for k in ids], [])
                value_ids = sum([[ind for i, index in enumerate(indices[:, axis]) if index == k] for ind, k in enumerate(ids)], [])
                data = data[valid_ids]
                indices = indices[valid_ids]
                indices[:, axis] = np.array(value_ids)
                shape[axis] = ids.__len__()
            elif isinstance(ids, int):
                valid_ids = [i for i, index in enumerate(indices[:, axis]) if index == ids]
                data = data[valid_ids]
                indices = indices[valid_ids]
                indices[:, axis] = 0
                shape[axis] = 0
            else:
                raise ValueError
        valid_axes = [axis for axis in range(shape.__len__()) if shape[axis] != 0]
        indices = indices[:, valid_axes]
        shape = [shape[axis] for axis in valid_axes]

        return SparseTensor(data, indices, shape, sparse_merge=False)

    # ------------- basic operations -------------
    def __neg__(self): return SparseTensor(-self.data, self.indices, self.shape, sparse_merge=False)

    def __add__(self, other):
        if type(other).__name__ in ["bool", "int", "float", "int64", "float64"]:
            return SparseTensor(self.data + other, self.indices, self.shape)
        elif isinstance(other, SparseTensor):
            assert sum([sax != oax for sax, oax in zip(self.shape, other.shape)]) == 0
            return SparseTensor(np.hstack([self.data, other.data]), np.vstack([self.indices, other.indices]), self.shape)
        else:
            raise ValueError

    def __sub__(self, other): return self.__add__(-other)

    def __mul__(self, other):
        if type(other).__name__ in ["bool", "int", "float"]:
            return SparseTensor(other * self.data, self.indices, self.shape, sparse_merge=False)

    @classmethod
    def reduce_sum(cls, *tensors):
        data = np.hstack([tensor.data for tensor in tensors])
        indices = np.vstack([tensor.indices for tensor in tensors])
        return SparseTensor(data, indices, tensors[0].shape)

    # ------------- shape operations -------------
    @classmethod
    def integer2base(cls, n, base):
        if not isinstance(n, list):
            n = [n]
        for b in list(base)[::-1]:
            n = [n[-1] % b] + n[:-1] + [n[-1] // b]
        return np.stack(n[:-1], axis=-1)

    @classmethod
    def base2integer(cls, n, base):
        shape = n.shape
        n = np.reshape(n, newshape=(-1, shape[-1]))
        integer = n[:, 0]
        for i in range(1, shape[-1]):
            integer = integer * base[i] + n[:, i]
        return np.reshape(integer, newshape=shape[:-1])

    def reshape(self, shape):
        def prod(seq): return reduce(lambda x, y: x * y, seq)
        shape = [axis if axis != -1 else -prod(self.shape) // prod(shape) for axis in shape]

        flatten_indices = SparseTensor.base2integer(self.indices, self.shape)
        indices = SparseTensor.integer2base(flatten_indices, shape)
        return SparseTensor(self.data, indices, shape, sparse_merge=False)

    def transpose(self, axes):
        return SparseTensor(self.data, self.indices[:, axes], [self.shape[axis] for axis in axes], sparse_merge=False)

    # ------------- Einstein summation convention -------------
    def __matmul__(self, other):
        """
        self: (M_1, ..., M_m, L) array_like
        other: (N_1, ..., N_n) array_like
        Return: (M_1, ..., M_m, N_1, ..., N_n) array_like
        """
        assert isinstance(other, np.ndarray) and other.shape.__len__() == 1
        data = self.data * other[self.indices[:, -1]]
        return SparseTensor(data, self.indices[:, :-1], self.shape[:-1], sparse_merge=False)

    @classmethod
    def einsum(cls, subscripts, *operands):
        inputs, output = subscripts.split('->')
        inputs = inputs.split(',')

        if operands.__len__() == 1:
            assert inputs[0].__len__() == set([s for s in inputs[0]]).__len__()
            axes = [inputs[0].index(s) for s in output]
            shape = [operands[0].shape[axis] for axis in axes]
            sparse_merge = shape.__len__() != operands[0].shape.__len__()
            return SparseTensor(operands[0].data, operands[0].indices[:, axes], shape, sparse_merge=sparse_merge)

        if operands.__len__() == 2:
            def prod(seq): return reduce(lambda x, y: x * y, seq)

            script_i = ''.join(sorted(list(set(list(inputs[0])) - set(list(inputs[1])))))
            script_j = ''.join(sorted(list(set(list(inputs[0])) - set(list(script_i)))))
            axes = [inputs[0].index(s) for s in script_i + script_j]
            inputs_ij = operands[0].transpose(axes)
            n_i = prod([inputs_ij.shape[axis] for axis in range(script_i.__len__())])
            n_j = prod([inputs_ij.shape[axis] for axis in range(script_i.__len__(), (script_i + script_j).__len__())])
            inputs_ij = inputs_ij.reshape([n_i, n_j])

            script_k = ''.join(sorted(list(set(list(inputs[1])) - set(list(inputs[0])))))
            axes = [inputs[1].index(s) for s in script_j + script_k]
            inputs_jk = operands[1].transpose(axes)
            n_k = prod([inputs_jk.shape[axis] for axis in range(script_j.__len__(), (script_j + script_k).__len__())])
            inputs_jk = inputs_jk.reshape([n_j, n_k])

            data, indices = [], []
            for dij, [i, j1] in zip(inputs_ij.data, inputs_ij.indices):
                for djk, [j2, k] in zip(inputs_jk.data, inputs_jk.indices):
                    if j1 == j2:
                        data.append(dij * djk)
                        indices.append([i, j1, k])
            inputs_ijk = cls(np.array(data), np.array(indices), [n_i, n_j, n_k])
            shape = [
                operands[0].shape[inputs[0].index(s)] if s in inputs[0] else operands[1].shape[inputs[1].index(s)]
                for s in script_i + script_j + script_k]
            inputs_ijk = inputs_ijk.reshape(shape)
            return cls.einsum(script_i + script_j + script_k + '->' + output, inputs_ijk)

        if operands.__len__() > 2:
            reduce_script = set(list(''.join(inputs[:2]))) - set(list(''.join(inputs[2:]))) - set(list(output))
            reduce_output = ''.join(sorted(list(set(list(''.join(inputs[:2]))) - reduce_script)))
            tensor = cls.einsum("{},{}->{}".format(inputs[0], inputs[1], reduce_output), operands[0], operands[1])
            operands = [tensor] + list(operands)[2:]
            return cls.einsum(','.join([reduce_output] + inputs[2:]) + '->' + output, *operands)


if __name__ == "__main__":
    tensor_1 = SparseTensor(
        data=np.random.rand(32, ),
        indices=np.random.randint(0, 3, size=(32, 3)),
        shape=(4, 3, 5)
    )
    error = tensor_1.dense()[:, [1, 2], 2] - tensor_1[:, [1, 2], 2].dense()
    print("check tensor_1[:, [1, 2], 2]:", np.max(np.abs(error)))

    tensor_2 = SparseTensor(
        data=np.random.rand(32, ),
        indices=np.random.randint(0, 3, size=(32, 3)),
        shape=(4, 3, 5)
    )
    error = (tensor_1 + tensor_2).dense() - (tensor_1.dense() + tensor_2.dense())
    print("check tensor_1 + tensor_2:", np.max(np.abs(error)))
    error = (tensor_1 - tensor_2).dense() - (tensor_1.dense() - tensor_2.dense())
    print("check tensor_1 - tensor_2:", np.max(np.abs(error)))

    error = tensor_1.reshape([5, -1, 6]).dense() - np.reshape(tensor_1.dense(), [5, 2, -1])
    print("check tensor_1.reshape:", np.max(np.abs(error)))

    tensor_1 = tensor_1.reshape([-1, 5])
    vector = tensor_2.reshape([5, -1])[:, 0].dense()
    error = (tensor_1@vector).dense() - tensor_1.dense()@vector
    print("check @:", np.max(np.abs(error)))

    tensor_1 = tensor_1.reshape([4, 3, 5])
    tensor_2 = tensor_2.reshape([4, 3, 5])
    error = SparseTensor.einsum("tij->tji", tensor_1).dense() - np.transpose(tensor_1.dense(), axes=[0, 2, 1])
    print("check einsum 'tij->tji':", np.max(np.abs(error)))
    error = SparseTensor.einsum("tij->tj", tensor_1).dense() - np.einsum("tij->tj", tensor_1.dense())
    print("check einsum 'tij->tj':", np.max(np.abs(error)))
    error = SparseTensor.einsum("tij->t", tensor_1).dense() - np.einsum("tij->t", tensor_1.dense())
    print("check einsum 'tij->t':", np.max(np.abs(error)))

    tensor_1 = tensor_1.reshape([4, 5, 3])
    tensor_2 = tensor_2.reshape([4, 3, 5])
    error = SparseTensor.einsum("tij,tki->tjk", tensor_1, tensor_2).dense() - np.einsum("tij,tki->tjk", tensor_1.dense(), tensor_2.dense())
    print("check einsum 'tij,tki->tjk':", np.max(np.abs(error)))

    tensor_1 = tensor_1.reshape([2, 2, 5, 3])
    tensor_2 = tensor_2.reshape([2, 3, 5, 2])
    error = SparseTensor.einsum("tpij,tkip->tjk", tensor_1, tensor_2).dense() - np.einsum("tpij,tkip->tjk", tensor_1.dense(), tensor_2.dense())
    print("check einsum 'tpij,tkip->tjk':", np.max(np.abs(error)))

    tensor_1 = tensor_1.reshape([5, 3, 2, 2])
    tensor_2 = tensor_2.reshape([5, 4, 3])
    tensor_3 = tensor_2.reshape([5, 6, 2])
    error = SparseTensor.einsum("tijk,tai,tbj,tck->tabc", tensor_1, tensor_2, tensor_3, tensor_3).dense() - \
            np.einsum("tijk,tai,tbj,tck->tabc", tensor_1.dense(), tensor_2.dense(), tensor_3.dense(), tensor_3.dense())
    print("check einsum 'tijk,tai,tbj,tck->tabc':", np.max(np.abs(error)))
