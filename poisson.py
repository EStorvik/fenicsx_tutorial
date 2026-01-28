# Fix MPI/OFI finalization errors on macOS
import os

os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"


from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, plot, io, la

from dolfinx.fem.petsc import LinearProblem

import ufl

import pyvista

# Mesh
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(16, 16),
    cell_type=mesh.CellType.triangle,
)

# Function space
V = fem.functionspace(msh, ("CG", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)


# Dicichlet boundary conditions
def marker(x):
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 1.0)
    )


facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1), marker=marker)
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# RHS function
x = ufl.SpatialCoordinate(msh)
f = -2.0 * (x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1]))

u_exact = x[0] * x[1] * (1.0 - x[0]) * (1.0 - x[1])

# Variational form
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Problem
problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options_prefix="test_poisson_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    },
)

uh = problem.solve()


l2_error = np.sqrt(
    fem.assemble_scalar(fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx))
)


print(f"Manual norm: {l2_error}")


###############################
# Plotting with matplotlib
###############################
coords = msh.geometry.x[:, :2]  # Get x, y coordinates
u_array = uh.x.array.real
triangles = msh.topology.connectivity(2, 0).array.reshape(-1, 3)

fig, ax = plt.subplots()
# Use tricontourf for filled contours
contour = ax.tricontourf(
    coords[:, 0], coords[:, 1], triangles, u_array, levels=20, cmap="viridis"
)

# Optionally add mesh lines
ax.triplot(coords[:, 0], coords[:, 1], triangles, "k-", lw=0.5, alpha=0.5)
ax.set_title("2D DOLFINx Function with Matplotlib")
fig.colorbar(contour, ax=ax)
plt.tight_layout()
# Use plt.show() to display and block until the window is closed
plt.show()

###############################
# Plotting with pyvista
###############################
cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
warped = grid.warp_by_scalar()
plotter.add_mesh(warped)
plotter.show()


with io.XDMFFile(msh.comm, "output/poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
