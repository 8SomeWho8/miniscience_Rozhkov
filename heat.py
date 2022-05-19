from fenics import *
import numpy as np

T = 0.25            # final time
num_steps = 600    # number of time steps
dt = T / num_steps  # time step size
a = 5
omega = 2*np.pi
T_1 = 1e4
T_2 = 2.5*1e4

# Create mesh and define function space
nx = ny = 200
mesh = RectangleMesh(Point(-2, -2), Point(2, 2), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('T_1*exp(-a*pow(x[0] - 0.5, 2) - a*pow(x[1] - 0.5, 2)) + T_2*exp(-a*pow(x[0] + 0.5, 2) - a*pow(x[1] + 0.5, 2))', 
                 degree=2, a=a, T_1=T_1, T_2=T_2)
#u_D = Expression('a*sin(omega*x[0])', degree=2, a=a, omega=omega)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
u_n = interpolate(u_D, V)
#u_n = project(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

res_file = File('heat/solution.pvd')

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bc)

    # Save solution to VTK
    res_file << u

    # Update previous solution
    u_n.assign(u)
