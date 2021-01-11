from defcon import *
from firedrake import *
import numpy
from petsc4py import PETSc

from firedrake import DistributedMeshOverlapType
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

num_threads = 20
#lusolver = "mkl_pardiso"
lusolver = "mumps"

class SmecticProblem(BifurcationProblem):
    def mesh(self, comm):
        self.levels = 0
        self.nviz = 1

        self.Nx = 6
        self.Ny = 6
        self.Nz = 5
        self.L = 3
        self.height = 2
        base = SquareMesh(self.Nx, self.Ny, self.L, comm=comm, quadrilateral=True)
        baseh = MeshHierarchy(base, self.levels + self.nviz)
        mh = ExtrudedMeshHierarchy(baseh, height=self.height, base_layer=self.Nz)
        # shift the mesh coordinates to [-1.5,1.5]x[-1.5,1.5]x[0,1]
        for mesh in mh:
            mesh.coordinates.dat.data[:,0] -= self.L/2
            mesh.coordinates.dat.data[:,1] -= self.L/2
        self.mh = mh
        self.CG = FunctionSpace(mh[-1], "CG", 3)
        return mh[self.levels]

    def function_space(self, mesh):
        U = FunctionSpace(mesh, "CG", 3)
        V = VectorFunctionSpace(mesh, "CG", 2, dim=5) #components for Q tensor
        Z  = MixedFunctionSpace([U, V])

        print("Z.dim(): %s %s" % (Z.dim(), [Z.sub(i).dim() for i in range(2)]))

        return Z

    def parameters(self):
        W = Constant(0)
        K = Constant(0)
        B = Constant(0)
        q = Constant(0)
        theta = Constant(0)
        return [(W, "W", "anchorweight"),
                (K, "K", r"$K$"),
                (B, "B", r"$B$"),
                (q, "q", r"$q$"),
                (theta, "theta", r"$\theta$")]

    def energy(self, z, params):
        q = params[3] #wavelength, so larger q gives smaller period, thus more number of stripes
        theta = params[4]  # theta: tilt angle at the top
        W = params[0] #larger W gives better alignment of the boundary conditions
        a = Constant(-5*2)
        b = Constant(0)
        c = Constant(5*2)
        K = params[1]
        B = params[2]
        l = Constant(30)

        s = FacetNormal(z.function_space().mesh())

        (u, d) = split(z)
        Q = as_tensor([[d[0], d[2], d[3]],
                       [d[2], d[1], d[4]],
                       [d[3], d[4], -(d[0] + d[1])]])

        I = as_matrix([[1,0,0],[0,1,0],[0,0,1]])
        mat = grad(grad(u)) + q**2 * (Q+I/3) * u
        x,y,_ = SpatialCoordinate(z.function_space().mesh())
        r2 = x**2+y**2+Constant(1e-15)

        Q_radial = as_matrix([[x**2/r2-1/3, x*y/r2, 0],
                              [x*y/r2, y**2/r2-1/3, 0],
                              [0, 0, -1/3]]) #radial inplane directors on bottom face

        Q_vertical = as_matrix([[-1/3, 0, 0],
                                [0, (sin(theta))**2-1/3, sin(theta)*cos(theta)],
                                [0, sin(theta)*cos(theta), (cos(theta))**2-1/3]])

        E = (
            + a/2 * u**2 * dx
            + b/3 * u**3 * dx
            + c/4 * u**4 * dx
            + B   * inner(mat, mat) * dx
            + K/2 * inner(grad(Q), grad(Q)) * dx
            - l/2 * tr(Q*Q) * dx
            - l/3 * tr(Q*Q*Q) * dx
            + l/2 * dot(tr(Q*Q), tr(Q*Q)) * dx #this form of bulk energy is now forcing s=1
            + W/2 * inner(Q-Q_radial, Q-Q_radial) * ds_b
            + W/2 * inner(Q-Q_vertical, Q-Q_vertical) * ds_t #top face: vertical
            )

        return E

    def lagrangian(self, z, params):
        E = self.energy(z, params)
        return E

    def residual(self, z, params, w):
        L = self.lagrangian(z, params)
        h = avg(CellDiameter(z.function_space().mesh()))
        (v,_) = split(w)
        (u,_) = split(z)
        n = FacetNormal(z.function_space().mesh())

        F = derivative(L, z, w) + h**(-3)*inner(jump(grad(u),n), jump(grad(v),n))*(dS_h+dS_v)
        return F

    def boundary_conditions(self, Z, params):
        return []

    def functionals(self):
        def energy(z, params):
            return assemble(self.energy(z, params))

        def indefEnergy(z, params):
            a = Constant(-5*2)
            b = Constant(0)
            c = Constant(5*2)
            (u, _) = split(z)
            j = assemble(
                + a/2 * u**2 * dx
                + b/3 * u**3 * dx
                + c/4 * u**4 * dx
            )
            return j

        def fourthEnergy(z, params):
            (u, d) = split(z)
            q = params[3]
            B = params[2]
            Q = as_tensor([[d[0], d[2], d[3]],
                           [d[2], d[1], d[4]],
                           [d[3], d[4], -(d[0] + d[1])]])
            I = as_matrix([[1,0,0],[0,1,0],[0,0,1]])
            mat = grad(grad(u)) + q**2 * (Q+I/3) * u
            j = assemble(B*inner(mat,mat)*dx)
            return j

        def elasticEnergy(z, params):
            (_, d) = split(z)
            K = params[1]
            Q = as_tensor([[d[0], d[2], d[3]],
                           [d[2], d[1], d[4]],
                           [d[3], d[4], -(d[0] + d[1])]])
            j = assemble(K/2 * inner(grad(Q), grad(Q)) *dx)
            return j

        def bulkEnergy(z, params):
            (_, d) = split(z)
            l = Constant(30)
            Q = as_tensor([[d[0], d[2], d[3]],
                           [d[2], d[1], d[4]],
                           [d[3], d[4], -(d[0] + d[1])]])
            j = assemble(-l/2 * tr(Q*Q) *dx - l/3* tr(Q*Q*Q)*dx + l/2 * dot(tr(Q*Q), tr(Q*Q))*dx )
            return j

        def anchoringEnergy(z, params):
            (_, d) = split(z)
            W = params[0]
            theta = params[4]
            s = FacetNormal(z.function_space().mesh())
            I = as_matrix([[1,0,0],[0,1,0],[0,0,1]])
            Q = as_tensor([[d[0], d[2], d[3]],
                           [d[2], d[1], d[4]],
                           [d[3], d[4], -(d[0] + d[1])]])
            x,y,_ = SpatialCoordinate(z.function_space().mesh())
            r2 = x**2+y**2+Constant(1e-15)
            Q_radial = as_matrix([[x**2/r2-1/3, x*y/r2, 0],
                                  [x*y/r2, y**2/r2-1/3, 0],
                                  [0, 0, -1/3]]) #radial inplane directors on bottom face
            Q_vertical = as_matrix([[-1/3, 0, 0],
                                    [0, (sin(theta))**2-1/3, sin(theta)*cos(theta)],
                                    [0, sin(theta)*cos(theta), (cos(theta))**2-1/3]])
            j = assemble(W/2 * inner(Q-Q_radial, Q-Q_radial) * ds_b + W/2 * inner(Q-Q_vertical,Q-Q_vertical) * ds_t)
            return j

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
                (anchoringEnergy, "anchoringEnergy", r"$E_5$"),
                (sqL2, "sqL2", r"$\|u\|^2$")
               ]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess_interpolate(self, Z, params, n):
        # Interpolate TFCD data as initial guess
        import vtktools
        vtu = vtktools.vtu("data/solution-0_0.vtu")

        V = FunctionSpace(Z.mesh(), "Lagrange", 3)
        W = VectorFunctionSpace(Z.mesh(), V.ufl_element(), dim=3)
        uCG = Function(V)
        dCG = Function(W)

        lu = {"ksp_type": "preonly",
              "pc_type": "lu",
              "mat_type": "aij",
              "pc_factor_mat_solver_type": lusolver,
              "mat_mkl_pardiso_65": num_threads,
              "mat_mumps_icntl_14": 200,}

        density = lambda X: vtu.ProbeData(numpy.c_[X, numpy.zeros(X.shape[0])], "Density").reshape((-1,))
        director = lambda X: vtu.ProbeData(numpy.c_[X, numpy.zeros(X.shape[0])], "director")[:, :]

        (x, y, z_axis) = SpatialCoordinate(W.ufl_domain())

        X = interpolate(as_vector([x, y, z_axis]), W)
        uCG.dat.data[:] = density(X.dat.data_ro)
        dCG.dat.data[:,:] = director(X.dat.data_ro)
        d1,d2,d3 = dCG[0], dCG[1], dCG[2]
        qCG = as_vector([d1**2-1/3, d2**2-1/3, d1*d2, d1*d3, d2*d3])
        z = Function(Z)
        z.sub(0).project(uCG, solver_parameters=lu)
        z.sub(1).project(qCG, solver_parameters=lu)

        return z

    def initial_guess(self, Z, params, n):
        # Tim's form for TFCD
        (x, y, z) = SpatialCoordinate(Z.mesh())
        R = Constant(1.5)
        denomz = sqrt(R**2+x**2+y**2-2*R*sqrt(x**2+y**2)+z**2)+Constant(1e-10)
        denomxy = (sqrt(x**2+y**2)+Constant(1e-10)) * denomz
        nx = x*(sqrt(x**2+y**2)-R)/denomxy
        ny = y*(sqrt(x**2+y**2)-R)/denomxy
        nz = z/denomz
        q0 = conditional(x**2+y**2>R**2, as_vector([-1/3, -1/3, 0, 0, 0]),
                as_vector([nx**2-1/3, ny**2-1/3, nx*ny, nx*nz, ny*nz]))
        #q0 = as_vector([nx**2-1/3, ny**2-1/3, nx*ny, nx*nz, ny*nz])

        lu = {"ksp_type": "preonly",
              "pc_type": "lu",
              "mat_type": "aij",
              "pc_factor_mat_solver_type": lusolver,
              "mat_mkl_pardiso_65": num_threads,
              "mat_mumps_icntl_14": 200,}

        zf = Function(Z)
        zf.sub(0).project(cos(6*pi*z), solver_parameters=lu)
        zf.sub(1).interpolate(q0)

        return zf

    def initial_guess_radial(self, Z, params, n):
        (x, y, z) = SpatialCoordinate(Z.mesh())
        r2 = x**2+y**2+z**2+Constant(1e-10)
        radial = as_vector([x**2/r2 - 1/3, y**2/r2 - 1/3,
                            x*y/r2, x*z/r2, y*z/r2])

        lu = {"ksp_type": "preonly",
              "pc_type": "lu",
              "mat_type": "aij",
              "pc_factor_mat_solver_type": lusolver,
              "mat_mkl_pardiso_65": num_threads,
              "mat_mumps_icntl_14": 200,}

        z = Function(Z)
        z.sub(0).project(Constant(1), solver_parameters=lu)
        z.sub(1).interpolate(radial)

        return z

    def number_solutions(self, params):
        return float("inf")

    def solver_parameters(self, params, task, **kwargs):
        if isinstance(task, DeflationTask):
            damping = 0.9
            maxits = 1000
        else:
            damping = 1.0
            maxits = 500

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
            #"ksp_view_mat": "draw",
            #"draw_pause": "-1",
            #"draw_save": "sparsity.ppm",
            #"draw_size": "1024,768",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": lusolver,
            "mat_mkl_pardiso_65": num_threads,
            "mat_mumps_icntl_14": 200,
        }
        return params

    def save_pvd(self, z, pvd, params):
        zsrc = z
        mesh = z.function_space().mesh()

        if self.nviz > 0:
            ele = z.function_space().ufl_element()
            transfer = TransferManager()
            for i in range(self.nviz):
                mesh = self.mh[self.levels + i + 1]
                Z = FunctionSpace(mesh, ele)
                znew = Function(Z)
                transfer.prolong(zsrc, znew)
                zsrc = znew

        (u, d) = zsrc.split()
        uv = project(u, self.CG)
        uv.rename("Density")

        #visualize the director
        d1,d2,d3,d4,d5 = d[0],d[1],d[2],d[3],d[4]
        Q = interpolate(as_tensor(((d1, d3, d4), (d3, d2, d5), (d4, d5, -(d1+d2)))), TensorFunctionSpace(mesh, "CG", 1))
        eigs, eigv = numpy.linalg.eigh(numpy.array(Q.vector()))
        beta = 1- 6*(tr(Q*Q*Q))**2 / (tr(Q*Q))**3
        beta = interpolate(beta, FunctionSpace(mesh, "CG", 1))
        beta.rename("biaxiality-parameter")
        s = Function(FunctionSpace(mesh, "CG", 1))
        s.interpolate(sqrt(3/2*tr(Q*Q)))
        s.rename("order-parameter")
        n = Function(VectorFunctionSpace(mesh, "CG", 1, dim=3))
        n.vector()[:,:] = eigv[:,:,2]
        n.rename("director")

        pvd.write(uv, n, s, beta)

    def save_h5(self, z, hdf):
        hdf.write(z.sub(0), 'density')
        hdf.write(z.sub(1), 'q')

    def monitor(self, params, branchid, solution, functionals):
        filename = "output/pvd/theta-%s-q-%s-K-%s-B-%s/solution-%d.pvd" % (params[4], params[3], params[1], params[2], branchid)
        pvd = File(filename, comm=solution.function_space().mesh().comm)
        self.save_pvd(solution, pvd, params)
        print("Wrote to %s" % filename)

if __name__ == "__main__":
    from mpi4py import MPI
    dc = DeflatedContinuation(problem=SmecticProblem(), teamsize=7, verbose=True, profile=False, clear_output=True, logfiles=True)
    dc.run(values={"W": 10, "K": 0.03, "B": 1e-3, "q": 10, "theta": linspace(0,pi/6,61)}, freeparam="theta")
