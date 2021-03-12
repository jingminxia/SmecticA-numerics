# -*- coding: utf-8 -*-
from firedrake import *
from defcon import *
from petsc4py import PETSc
import numpy as np

class SmecticProblem(BifurcationProblem):
    def mesh(self, comm):

        baseN = 30
        base = PeriodicRectangleMesh(3*baseN, baseN, 1, 2, direction="x", comm=comm, quadrilateral=True)
        base.coordinates.dat.data[:,0] -= 0.5
        self.CG = FunctionSpace(base, "CG", 3)
        self.orig_coordinates = Function(base.coordinates)

        return base

    def function_space(self, mesh):
        U = FunctionSpace(mesh, "CG", 3)
        V = VectorFunctionSpace(mesh, "CG", 2, dim=2)
        Z  = MixedFunctionSpace([U, V])

        print("Z.dim(): %s %s" % (Z.dim(), [Z.sub(i).dim() for i in range(2)]))

        return Z

    def parameters(self):
        q = Constant(0)
        ratio = Constant(0)
        theta = Constant(0)
        return [(q, "q", r"$q$"),
                (ratio, "ratio", "aspect-ratio"),
                (theta, "theta", r"$\theta$")]

    def energy(self, z, params):
        q = params[0]
        r = params[1]
        theta = params[2]
        a = Constant(-5*2)
        b = Constant(0)
        c = Constant(5*2)
        B = Constant(1e-5)
        K = Constant(0.3)
        l = Constant(30)

        (u, d) = split(z)
        Q = as_tensor([[d[0], d[1]],
                       [d[1], -d[0]]])
        I = as_matrix([[1,0],[0,1]])
        mat = grad(grad(u)) + q**2 * (Q+I/2) * u
        E = (
            + a/2 * u**2 * dx
            + b/3 * u**3 * dx
            + c/4 * u**4 * dx
            + B   * inner(mat, mat) * dx
            + K/2 * inner(grad(Q), grad(Q)) * dx
            - l * tr(Q*Q) * dx
            + l * dot(tr(Q*Q), tr(Q*Q)) * dx
            )

        return E

    def lagrangian(self, z, params):
        E = self.energy(z, params)
        return E

    def residual(self, z, params, w):
        L = self.lagrangian(z, params)
        h = avg(CellDiameter(z.function_space().mesh()))
        s = FacetNormal(z.function_space().mesh())
        (u,_) = split(z)
        (v,_) = split(w)
        F = derivative(L, z, w) + h**(-3)*inner(jump(grad(u),s), jump(grad(v),s))*dS
        return F

    def boundary_conditions(self, Z, params):
        theta = params[2]
        bc1 = DirichletBC(Z.sub(1), as_vector([cos(theta)**2-1/2, -sin(theta)*cos(theta)]), 1)#bottom
        bc2 = DirichletBC(Z.sub(1), as_vector([cos(theta)**2-1/2, sin(theta)*cos(theta)]), 2)#top
        bcs = [bc1, bc2]
        return bcs

    def functionals(self):
        def energy(z, params):
            r = params[1]
            return assemble(self.energy(z, params)) /r

        def indefEnergy(z, params):
            r = params[1]
            a = Constant(-5*2)
            b = Constant(0)
            c = Constant(5*2)
            (u, _) = split(z)
            j = assemble(
                + a/2 * u**2 * dx
                + b/3 * u**3 * dx
                + c/4 * u**4 * dx
            )
            return j/r

        def fourthEnergy(z, params):
            (u, d) = split(z)
            q = params[0]
            r = params[1]
            B = Constant(1e-5)
            Q = as_tensor([[d[0], d[1]],
                           [d[1], -d[0]]])
            I = as_matrix([[1,0],[0,1]])
            mat = grad(grad(u)) + q**2 * (Q+I/2) * u
            j = assemble(B*inner(mat,mat)*dx)
            return j/r

        def elasticEnergy(z, params):
            (_, d) = split(z)
            K = Constant(0.3)
            r = params[1]
            Q = as_tensor([[d[0], d[1]],
                           [d[1], -d[0]]])
            j = assemble(K/2 * inner(grad(Q), grad(Q)) *dx)
            return j/r

        def bulkEnergy(z, params):
            (_, d) = split(z)
            r = params[1]
            l = Constant(30)
            Q = as_tensor([[d[0], d[1]],
                           [d[1], -d[0]]])
            j = assemble(-l * tr(Q*Q) *dx + l * dot(tr(Q*Q), tr(Q*Q))*dx )
            return j/r

        def sqL2(z, params):
            (u, d) = split(z)
            j = assemble(u*u*dx)
            return j

        return [
                (energy, "energy", r"$E(u, d)$"),
                (indefEnergy, "indefEnergy", r"$E_1$"),
                (fourthEnergy, "fourthEnergy", r"$E_2$"),
                (elasticEnergy, "elasticEnergy", r"$E_3$"),
                (bulkEnergy, "bulkEnergy", r"$E_4$"),
                (sqL2, "sqL2", r"$\|u\|^2$")
               ]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        # Tim's form for TFCD
        (x, y) = SpatialCoordinate(Z.mesh())
        R = Constant(0.5)
        denomy = sqrt(R**2+x**2-2*R*sqrt(x**2)+y**2)+Constant(1e-10)
        denomx = (sqrt(x**2)+Constant(1e-10)) * denomy
        nx = x*(sqrt(x**2)-R)/denomy
        ny = y/denomy
        q0 = conditional(x**2>R**2, as_vector([-1/2, 0]),
                as_vector([nx**2-1/2, nx*ny]))

        lu = {"ksp_type": "preonly",
              "pc_type": "lu",
              "mat_type": "aij",
              "pc_factor_mat_solver_type": "mumps",
              "mat_mumps_icntl_14": 200,}

        z = Function(Z)
        z.sub(0).project(Constant(1.0), solver_parameters=lu)
        z.sub(1).interpolate(q0)

        return z

    def initial_guess_radial(self, Z, params, n):
        r = params[1]
        z = Function(Z)
        (x, y) = SpatialCoordinate(Z.mesh())
        r2 = x**2+y**2 + Constant(1e-15)
        radial = as_vector([x**2/r2 -1/2, x*y/r2])

        lu = {"ksp_type": "preonly",
              "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps",
              "mat_mumps_icntl_14": 200}

        z.split()[0].project(Constant(1.0), solver_parameters=lu)
        z.split()[1].interpolate(radial)

        return z

    def transform_guess(self, state, task, io):
        r = task.newparams[1]
        print("Setting aspect ratio to r = %s" % r)
        mesh = state.function_space().mesh()
        mesh.coordinates.dat.data[:, 0] = r * self.orig_coordinates.dat.data_ro[:, 0]
        mesh.coordinates.dat.data[:, 1] = self.orig_coordinates.dat.data_ro[:, 1]

    def number_solutions(self, params):
        return float("inf")

    def solver_parameters(self, params, task, **kwargs):
        if isinstance(task, DeflationTask):
            damping = 0.9
            maxits = 1000
        else:
            damping = 1.0
            maxits = 1000

        params = {
            "snes_max_it": maxits,
            "snes_atol": 1.0e-8,
            "snes_rtol": 1.0e-8,
            "snes_monitor": None,
            "snes_linesearch_type": "l2",
            "snes_linesearch_monitor": None,
            "snes_linesearch_maxstep": 1.0,
            "snes_linesearch_damping": damping,
            "snes_converged_reason": None,
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200,
            "mat_mumps_icntl_24": 1,
            "mat_mumps_icntl_13": 1,
            "tao_type": "bnls",
            "tao_gmonitor": None,
            "tao_grtol": 1e-10,
            #"tao_monitor": None,
            #"tao_ls_monitor": None,
            "tao_bnk_ksp_type": "gmres",
            "tao_bnk_ksp_converged_reason": None,
            "tao_bnk_ksp_monitor_true_residual": None,
            "tao_bnk_pc_type": "lu",
            "tao_ntr_pc_factor_mat_solver_type": "mumps",
            #"tao_ntr_pc_type": "lmvm",
            "tao_ls_type": "armijo",
            "tao_converged_reason": None,
        }
        return params

    def save_pvd(self, z, pvd, params):
        mesh = z.function_space().mesh()
        r = params[1]
        mesh.coordinates.dat.data[:, 0] = r * self.orig_coordinates.dat.data_ro[:, 0]
        mesh.coordinates.dat.data[:, 1] = self.orig_coordinates.dat.data_ro[:, 1]

        (u, d) = z.split()
        uv = project(u, self.CG)
        uv.rename("Density")

        #visualize the director
        d0 = d[0]
        d1 = d[1]
        Q = interpolate(as_tensor([[d0, d1], [d1, -d0]]), TensorFunctionSpace(mesh, "CG", 1))
        eigs, eigv = np.linalg.eigh(np.array(Q.vector()))
        s = Function(FunctionSpace(mesh, "CG", 1)).interpolate(2*sqrt(dot(d,d)))
        s.rename("order-parameter")
        s_eig = Function(FunctionSpace(mesh, "CG", 1))
        s_eig.vector()[:] = 2*eigs[:,1]
        s_eig.rename("order-parameter-via-eig")
        n = Function(VectorFunctionSpace(mesh, "CG", 1))
        n.vector()[:,:] = eigv[:,:,1]
        n.rename("Director")

        pvd.write(uv, n, s, s_eig)

    def monitor(self, params, branchid, solution, functionals):
        filename = "output/pvd/theta-%s/solution-%d.pvd" % (params[2], branchid)
        pvd = File(filename, comm=solution.function_space().mesh().comm)
        self.save_pvd(solution, pvd, params)
        print("Wrote to %s" % filename)

    def compute_stability(self, params, branchid, z, hint=None):
        Z = z.function_space()
        trial = TrialFunction(Z)
        test  = TestFunction(Z)

        bcs = self.boundary_conditions(Z, params)
        comm = Z.mesh().mpi_comm()

        F = self.residual(z, [Constant(p) for p in params], test)
        J = derivative(F, z, trial)

        # Build the LHS matrix
        A = assemble(J, bcs=bcs, mat_type="aij")
        A = A.M.handle

        pc = PETSc.PC().create(comm)
        pc.setOperators(A)
        pc.setType("cholesky")
        pc.setFactorSolverType("mumps")
        pc.setUp()

        F = pc.getFactorMatrix()
        (neg, zero, pos) = F.getInertia()

        print("Inertia: (-: %s, 0: %s, +: %s)" % (neg, zero, pos))
        expected_dim = 0

        # Nocedal & Wright, theorem 16.3
        if neg == expected_dim:
            is_stable = True
        else:
            is_stable = False

        d = {"stable": (neg, zero, pos)}
        return d

params = linspace(0,pi/2,201)

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=SmecticProblem(), teamsize=3, verbose=True, profile=False, clear_output=True, logfiles=True)
    dc.run(values={"q": 30, "ratio": 4.0, "theta": params}, freeparam="theta")
