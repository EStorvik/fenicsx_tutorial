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
import time

import pyvista
import pyvistaqt

# Mesh
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(64, 64),
    cell_type=mesh.CellType.triangle,
)

dt = 0.1
num_time_steps = 100

# Function space
V = fem.functionspace(msh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

u_old = fem.Function(V)
u_old.x.array[:] = 0.0



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

# Create a Constant for time that can be updated
t = fem.Constant(msh, ScalarType(0))

# Use the constant in the expressions
f = 2.0 * (x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1])) * t + x[0] * x[1] * (1.0 - x[0]) * (1.0 - x[1])
u_exact = x[0] * x[1] * (1.0 - x[0]) * (1.0 - x[1]) * t

# Variational form
a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = dt * ufl.inner(f, v) * ufl.dx + ufl.inner(u_old, v) * ufl.dx

# Problem
problem = NonlinearProblem(
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


###############################
# Plotting with pyvista
###############################
uh = problem.solve()

print(type(uh))


cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["uh"] = uh.x.array[:].real
grid.set_active_scalars("uh")

p = pyvistaqt.BackgroundPlotter(title="u", auto_update=True)
p.add_mesh(grid, clim =[0, 10*0.5**4])
p.view_xy(True)
p.add_text(f"time: {t.value}", font_size=12, name="timelabel")



for i in range(num_time_steps):
    
    t.value += dt  # Update the time constant
    
    uh = problem.solve()
    uh.x.scatter_forward()
    u_old.x.array[:] = uh.x.array[:]
    u_old.x.scatter_forward()
    l2_error = np.sqrt(
    fem.assemble_scalar(fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx))
    )
    print(l2_error)
    # Add scalar data to grid
    p.add_text(f"time: {t.value:.2e}", font_size=12, name="timelabel")
    grid.point_data["uh"] = uh.x.array[:].real
    p.app.processEvents()
    time.sleep(0.1)



l2_error = np.sqrt(
    fem.assemble_scalar(fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx))
)








with io.XDMFFile(msh.comm, "output/poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
