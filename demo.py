import time
import numpy as np

from mesh import IsotropicMesh
from interpolation import BasisFunctions, LagrangeBasisFunctions

from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# TODO: * adding non-linear PDEs and high order examples
# TODO: * adding draw method for showing high dimensional situation.


def numerical_gradients(func, x, eps=1e-6):
    dim = x.shape[-1]
    gradients = []
    for d in range(dim):
        vec = np.eye(dim)[[d]]
        gradients.append((func(x + eps * vec) - func(x - eps * vec)) / (2 * eps))
    return np.stack(gradients, axis=1)


def numerical_laplace(func, x, eps=1e-6):
    dim = x.shape[-1]
    second_derivative = 0
    for d in range(dim):
        vec = np.eye(dim)[[d]]
        second_derivative = second_derivative + (func(x + eps * vec) - 2 * func(x) + func(x - eps * vec)) / eps ** 2
    return second_derivative


def cube_region(num=3):
    from scipy.spatial import Delaunay

    def refine_mesh(points):
        nn, dim = points.shape
        edges = Delaunay(points).simplices[:, [[i, j] for i in range(1, dim + 1) for j in range(i)]]  # [NT, NE, 2]
        row = np.minimum(edges[:, :, 0].flatten(), edges[:, :, 1].flatten())
        col = np.maximum(edges[:, :, 0].flatten(), edges[:, :, 1].flatten())
        indices = np.unique(np.stack([row, col], axis=1), axis=0)
        points = np.vstack([points, np.mean(points[indices, :], axis=1)])
        return np.unique(points, axis=0)

    nodes = np.vstack([-1 + 2 * np.array([[i // 4, i % 4 // 2, i % 2] for i in range(8)], dtype=np.float), np.array([[0., 0., 0.]])])
    for _ in range(num):
        nodes = refine_mesh(nodes)
    return nodes


def cassini_oval_region(num=16):
    """(x^2 + y^2)^2 - 2(x^2 - y^2) = 3"""
    from skimage import measure

    # generate dense points in bound
    phi = np.linspace(0, 2 * np.pi, num=10000, endpoint=False)
    r = np.sqrt(np.cos(2 * phi) + np.sqrt(4 - np.sin(2 * phi) ** 2))
    dense_boundary_points = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)

    # generate nodes in bound
    phi = np.linspace(0, 2 * np.pi, num=num, endpoint=False)
    r = np.sqrt(np.cos(2 * phi) + np.sqrt(4 - np.sin(2 * phi) ** 2))
    boundary_points = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)

    # generate nodes in interal region
    gap = np.min(np.linalg.norm(boundary_points[:-1] - boundary_points[1:], axis=1))
    xv, yv = np.meshgrid(np.arange(-2, 2, gap), np.arange(-1.5, 1.5, gap))
    unsafe_inner_points = np.stack([xv.flatten(), yv.flatten()], axis=1)
    masks = measure.points_in_poly(points=unsafe_inner_points, verts=boundary_points)
    unsafe_inner_points = unsafe_inner_points[[i for i, mask in enumerate(masks) if mask]]
    distance = np.linalg.norm(np.expand_dims(unsafe_inner_points, axis=1) - np.expand_dims(dense_boundary_points, axis=0), axis=-1)
    inner_points = unsafe_inner_points[[i for i, dis in enumerate(np.min(distance, axis=1)) if dis > 0.5 * gap]]

    return np.vstack([boundary_points, inner_points])


def example_3d(path=None, degree=3, num_refine=3):
    """
    Problem:
        -\Delta u + \boldsymbol{c} \cdot \nabla u + u^2 = f, \quad \boldsymbol{x} \in \Omega
        u = g, \quad \boldsymbol{x} \in \partial \Omega

    Combination of basis functions:
        u_h = \sum_i (p_i \psi_i), \quad \psi_i \in P_1(T_h)

    The weak form:
        \sum_j ((\nabla \psi_i, \nabla \psi_j) + (\psi_i, \boldsymbol{c} \cdot \nabla \psi_j) + (\psi_i, \psi_j)) p_i = (\psi_i, f)
    """
    dim = 3
    def func_u(x): return np.exp(-np.sum(np.square(x), axis=1)) + np.sum(np.cos(x), axis=1)
    def func_u_1(x): return -2 * np.sum(x, axis=1) * np.exp(-np.sum(np.square(x), axis=1)) - np.sum(np.sin(x), axis=1)
    def func_u_2(x): return (4 * np.sum(x ** 2, axis=1) - 2 * dim) * np.exp(-np.sum(np.square(x), axis=1)) - np.sum(np.cos(x), axis=1)
    def func_u_3(x): return (-8 * np.sum(x ** 3, axis=1) + 12 * np.sum(x, axis=1)) * np.exp(-np.sum(np.square(x), axis=1)) + np.sum(np.sin(x), axis=1)
    def func_f(x): return -func_u_2(x) + func_u_1(x) + func_u(x) ** 2
    def func_g(x): return func_u(x)

    # check PDEs
    def estimate_func_f(x): return -numerical_laplace(func_u, x) + np.sum(numerical_gradients(func_u, x), axis=1) + func_u(x) ** 2
    check_point = np.random.rand(1, dim)
    print("check:", func_f(check_point), estimate_func_f(check_point))

    # Start to solve
    mesh = IsotropicMesh(nodes=cube_region(num_refine))
    poly = LagrangeBasisFunctions(degree=degree, mesh=mesh)

    # --------------------------------------------------
    def parameters():
        roots = np.zeros(shape=(poly.domain_dof, ), dtype=np.float)
        coeff_g_tensor, = poly.exact2numerical(func_u_seq=[func_g, ])
        for t in range(poly.mesh.nt):
            for f in range(poly.simplex_dof):
                roots[poly.map[(t, f)]] = coeff_g_tensor[t, f]
        # (grad_pk, grad_pk)
        comb_x = poly.linear_combination(c=np.array([1, 0, 0]), order=1)
        comb_y = poly.linear_combination(c=np.array([0, 1, 0]), order=1)
        comb_z = poly.linear_combination(c=np.array([0, 0, 1]), order=1)
        dense_x = BasisFunctions.dense_integral(combinations=[comb_x, comb_x], mesh=mesh)
        dense_y = BasisFunctions.dense_integral(combinations=[comb_y, comb_y], mesh=mesh)
        dense_z = BasisFunctions.dense_integral(combinations=[comb_z, comb_z], mesh=mesh)
        sparse_1 = BasisFunctions.sparse_integral(dense_integral=dense_x + dense_y + dense_z, shape_functions_seq=[poly, poly])
        # (pk, c * grad_pk)
        comb = poly.linear_combination(c=1, order=0)
        comb_c = poly.linear_combination(c=np.array([1, 1, 1]), order=1)
        dense = BasisFunctions.dense_integral(combinations=[comb, comb_c], mesh=mesh)
        sparse_2 = BasisFunctions.sparse_integral(dense_integral=dense, shape_functions_seq=[poly, poly])
        # (pk, pk, pk)
        dense = BasisFunctions.dense_integral(combinations=[comb, comb, comb], mesh=mesh)
        sparse_3 = BasisFunctions.sparse_integral(dense_integral=dense, shape_functions_seq=[poly, poly, poly])
        # (pk, f)
        dense = poly.dense_numerical_integral(combination=comb, func=func_f, mesh=mesh)
        sparse_4 = BasisFunctions.sparse_numerical_integral(dense_numerical_integral=dense, shape_functions=poly)
        return roots, sparse_1[poly.inner_indices, :], sparse_2[poly.inner_indices, :], sparse_3[poly.inner_indices, :, :], sparse_4[poly.inner_indices]
    # --------------------------------------------------
    roots, sparse_1, sparse_2, sparse_3, rhs = parameters()
    print(sparse_3.shape)

    def fun(placeholder):
        start_time = time.time()  # TODO:
        roots[poly.inner_indices] = placeholder
        lhs = (sparse_1@roots + sparse_2@roots + sparse_3@roots@roots).dense()
        error = np.sum(np.square(lhs - rhs))
        print("error:", error)
        print("time consumption fun:", time.time() - start_time)  # TODO:
        return error

    def jacobi(placeholder):
        start_time = time.time()  # TODO:
        roots[poly.inner_indices] = placeholder
        temp = sparse_3@roots
        lhs = (sparse_1@roots + sparse_2@roots + temp@roots).dense()
        matrix = sparse_1 + sparse_2 + temp * 2
        jac = 2 * (matrix.transpose([1, 0])@(lhs - rhs)).dense()[poly.inner_indices]
        print("time consumption jac:", time.time() - start_time)  # TODO:
        print()
        return jac

    res = minimize(
        fun=fun,
        x0=np.zeros(shape=(poly.inner_indices.__len__(), )),
        method='BFGS',
        jac=jacobi,
        options={'maxiter': 1000, 'disp': True}
    )
    roots[poly.inner_indices] = res.x

    interpolation_points, exact_u, numerical_u, error = poly.error(roots, func_u, order=2)
    print("error:", error)

    print("min:", interpolation_points.min())
    print("max:", interpolation_points.max())

    # remove convex hull
    hull_sign = np.max(np.abs(interpolation_points), axis=1) != np.max(np.abs(interpolation_points))
    hull_indices = [i for i, sign in enumerate(hull_sign) if sign]
    interpolation_points = interpolation_points[hull_indices]
    exact_u = exact_u[hull_indices]
    numerical_u = numerical_u[hull_indices]

    # search new convex hull
    hull_sign = np.max(np.abs(interpolation_points), axis=1) == np.max(np.abs(interpolation_points))
    hull_indices = [i for i, sign in enumerate(hull_sign) if sign]
    interpolation_points = interpolation_points[hull_indices]
    exact_u = exact_u[hull_indices]
    numerical_u = numerical_u[hull_indices]

    # remove repetitive nodes
    hull_vertices, unique_indices = np.unique(interpolation_points, axis=0, return_index=True)
    exact_u = exact_u[unique_indices]
    numerical_u = numerical_u[unique_indices]

    # construct convex hull
    from scipy.spatial import Delaunay
    convex_hull = Delaunay(hull_vertices).convex_hull
    exact_u = np.mean(exact_u[convex_hull], axis=1)
    numerical_u = np.mean(numerical_u[convex_hull], axis=1)
    numerical_colors = np.array([[u, 0, 1 - u] for u in (numerical_u - np.min(numerical_u)) / (np.max(numerical_u) - np.min(numerical_u))])
    numerical_polygons = Poly3DCollection(hull_vertices[convex_hull, :], color=numerical_colors, alpha=0.5, linewidth=0)
    hull_errors = exact_u - numerical_u
    error_colors = np.array([[u, 0, 1 - u] for u in (hull_errors - np.min(hull_errors)) / (np.max(hull_errors) - np.min(hull_errors))])
    error_polygons = Poly3DCollection(hull_vertices[convex_hull, :], color=error_colors, alpha=0.5, linewidth=0)

    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_title("numerical solution".format(degree, poly.simplex_dof))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.add_collection(numerical_polygons)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.set_title(r"$L_{}$ error: {:.2e}".format(2, error))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.add_collection(error_polygons)

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.set_title("Lagrange form (degree={}, dof={})".format(degree, poly.simplex_dof))
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dof_points = np.einsum("vd,fv->fd", vertices, poly.dof_points)
    surfaces = np.stack([np.roll(vertices, k, axis=0)[1:, :] for k in range(dim + 1)], axis=0)
    ax.add_collection(Poly3DCollection(surfaces, alpha=0.5, edgecolor='b'))
    ax.scatter(dof_points[:, 0], dof_points[:, 1], dof_points[:, 2])
    for weights, shape_function in zip(dof_points, poly.shape_functions):
        subsrcipts = []
        for c, ds in zip(shape_function.coeff, shape_function.indices):
            sign = "" if c < 0 else "+".format(c)
            subsrcipts.append(sign + "{:.2f}".format(c) + ''.join(["$\lambda_{}^{}$".format(k, d) for k, d in enumerate(ds)]))
        ax.text(weights[0], weights[1], weights[2], '\n'.join(subsrcipts), fontsize=6)

    fig.savefig(path)


def example_2d(path=None, degree=3, num_refine=3):
    """
    Problem:
        -\Delta (\boldsymbol{c} \cdot \nabla u) + u^2 = f, \quad \boldsymbol{x} \in \Omega
        u = g, \quad \boldsymbol{x} \in \partial \Omega

    Combination of basis functions:
        u_h = \sum_i (p_i \psi_i^u), \quad \psi_i^u \in P_3(T_h)

    The weak form:
        \sum_j ((\nabla \psi_i^u, \nabla \psi_j^u) + (\psi_i^u, \psi_j^u)) p_j + \sum_k (\psi_i^u, \psi_k^v) q_k = (\psi_i^u, f)
        \sum_j (\psi_i^v, \boldsymbol{c} \cdot \nabla \psi_j^u) p_j - \sum_k (\psi_i^v, \psi_k^v) q_k = 0
    """
    dim = 2
    def func_u(x): return np.exp(-np.sum(np.square(x), axis=1)) + np.sum(np.cos(x), axis=1)
    def func_u_1(x): return -2 * np.sum(x, axis=1) * np.exp(-np.sum(np.square(x), axis=1)) - np.sum(np.sin(x), axis=1)
    def func_u_2(x): return (4 * np.sum(x ** 2, axis=1) - 2 * dim) * np.exp(-np.sum(np.square(x), axis=1)) - np.sum(np.cos(x), axis=1)
    def func_f(x): return -func_u_2(x) + func_u_1(x) + func_u(x) ** 2
    def func_g(x): return func_u(x)

    # check PDEs
    def estimate_func_f(x): return -numerical_laplace(func_u, x) + np.sum(numerical_gradients(func_u, x), axis=1) + func_u(x) ** 2
    check_point = np.random.rand(1, dim)
    print("check:", func_f(check_point), estimate_func_f(check_point))

    # Start to solve
    mesh = IsotropicMesh(nodes=cassini_oval_region(4 * 2 ** num_refine))
    poly = LagrangeBasisFunctions(degree=degree, mesh=mesh)

    # --------------------------------------------------
    def parameters():
        roots = np.zeros(shape=(poly.domain_dof, ), dtype=np.float)
        coeff_g_tensor, = poly.exact2numerical(func_u_seq=[func_g, ])
        for t in range(poly.mesh.nt):
            for f in range(poly.simplex_dof):
                roots[poly.map[(t, f)]] = coeff_g_tensor[t, f]
        # (grad_pk, grad_pk)
        comb_x = poly.linear_combination(c=np.array([1, 0]), order=1)
        comb_y = poly.linear_combination(c=np.array([0, 1]), order=1)
        dense_x = BasisFunctions.dense_integral(combinations=[comb_x, comb_x], mesh=mesh)
        dense_y = BasisFunctions.dense_integral(combinations=[comb_y, comb_y], mesh=mesh)
        sparse_1 = BasisFunctions.sparse_integral(dense_integral=dense_x + dense_y, shape_functions_seq=[poly, poly])
        # (pk, c * grad_pk)
        comb = poly.linear_combination(c=1, order=0)
        comb_c = poly.linear_combination(c=np.array([1, 1]), order=1)
        dense = BasisFunctions.dense_integral(combinations=[comb, comb_c], mesh=mesh)
        sparse_2 = BasisFunctions.sparse_integral(dense_integral=dense, shape_functions_seq=[poly, poly])
        # (pk, pk, pk)
        dense = BasisFunctions.dense_integral(combinations=[comb, comb, comb], mesh=mesh)
        sparse_3 = BasisFunctions.sparse_integral(dense_integral=dense, shape_functions_seq=[poly, poly, poly])
        # (pk, f)
        dense = poly.dense_numerical_integral(combination=comb, func=func_f, mesh=mesh)
        sparse_4 = BasisFunctions.sparse_numerical_integral(dense_numerical_integral=dense, shape_functions=poly)
        return roots, sparse_1[poly.inner_indices, :], sparse_2[poly.inner_indices, :], sparse_3[poly.inner_indices, :, :], sparse_4[poly.inner_indices]
    # --------------------------------------------------
    roots, sparse_1, sparse_2, sparse_3, rhs = parameters()

    def fun(placeholder):
        start_time = time.time()  # TODO:
        roots[poly.inner_indices] = placeholder
        lhs = (sparse_1@roots + sparse_2@roots + sparse_3@roots@roots).dense()
        error = np.sum(np.square(lhs - rhs))
        print("error:", error)
        print("time consumption fun:", time.time() - start_time)  # TODO:
        return error

    def jacobi(placeholder):
        start_time = time.time()  # TODO:
        roots[poly.inner_indices] = placeholder
        temp = sparse_3@roots
        lhs = (sparse_1@roots + sparse_2@roots + temp@roots).dense()
        matrix = sparse_1 + sparse_2 + temp * 2
        jac = 2 * (matrix.transpose([1, 0])@(lhs - rhs)).dense()[poly.inner_indices]
        print("time consumption jac:", time.time() - start_time)  # TODO:
        print()
        return jac

    res = minimize(
        fun=fun,
        x0=np.zeros(shape=(poly.inner_indices.__len__(), )),
        method='BFGS',
        jac=jacobi,
        options={'maxiter': 1000, 'disp': True}
    )
    roots[poly.inner_indices] = res.x
    interpolation_points, exact_u, numerical_u, error = poly.error(roots, func_u, order=2)

    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_title("numerical solution")
    ax.plot_trisurf(interpolation_points[:, 0], interpolation_points[:, 1], numerical_u, alpha=0.5)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.set_title(r"$L_{}$ error: {:.2e}".format(2, error))
    ax.plot_trisurf(interpolation_points[:, 0], interpolation_points[:, 1], exact_u - numerical_u, alpha=0.5)

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Lagrange form (degree={}, dof={})".format(degree, poly.simplex_dof))
    radian = np.linspace(0.5 * np.pi, 2.5 * np.pi, 3, endpoint=False)
    vertices = np.stack([np.cos(radian), np.sin(radian)], axis=1)
    dof_points = np.einsum("vd,fv->fd", vertices, poly.dof_points)
    ax.plot(vertices[[0, 1, 2, 0], 0], vertices[[0, 1, 2, 0], 1])
    ax.scatter(dof_points[:, 0], dof_points[:, 1])
    for weights, shape_function in zip(dof_points, poly.shape_functions):
        subsrcipts = []
        for c, ds in zip(shape_function.coeff, shape_function.indices):
            sign = "" if c < 0 else "+".format(c)
            subsrcipts.append(sign + "{:.2f}".format(c) + ''.join(["$\lambda_{}^{}$".format(k, d) for k, d in enumerate(ds)]))
        ax.text(weights[0], weights[1], '\n'.join(subsrcipts), fontsize=6)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

    fig.savefig(path)


if __name__ == "__main__":
    start_time = time.time()
    # example_3d(path="./3d.png", degree=2, num_refine=2)
    example_2d(path="./2d.png", degree=1, num_refine=3)
    print("time consumption:", time.time() - start_time)
    plt.show()
