from firedrake import *
from firedrake.petsc import PETSc
from firedrake.assemble import create_assembly_callable, allocate_matrix
from firedrake.preconditioners.patch import bcdofs
from slepc4py import SLEPc

from defcon import *
import defcon.backend as backend
from oilystreaks import SmecticProblem

class TaoOptimisation(object):
    def __init__(self, problem, params, branchid):
        self.params = params
        self.problem = problem
        comm = MPI.COMM_WORLD

        mesh = problem.mesh(comm)
        V = problem.function_space(mesh)

        z = Function(V)
        v = TestFunction(V)
        w = TrialFunction(V)
        self.z = z

        # Set PETSc options
        opts = problem.solver_parameters(params, None)
        popts = PETSc.Options()
        for key in opts:
            popts.setValue(key, opts[key])


        # Set initial guess
        outputdir = "output"
        self.io = problem.io(outputdir)
        functionals = problem.functionals()
        _params = problem.parameters()
        self.io.setup(_params, functionals, V)
        self.base = self.io.fetch_solutions(params, [branchid])[0]

        # print the original energy of the base solution
        functional_base = self.io.fetch_functionals([params], branchid)[0][0]
        print("The energy functional of the base solution: %s" % functional_base)

        # check the stability of the solution
        (_, perturbations) = self.compute_stability(self.base, problem)

        #pcg = PCG64(seed=123456789)
        #rg = RandomGenerator(pcg)
        #eps = 0e-10
        #noise = rg.uniform(V, -eps, +eps)

        #z.assign(1.1*self.base + 0.4*perturbations[0]) # used for r=4.6
        #z.assign(1.001*self.base + 0.1*perturbations[-2]) # used for r=4.4 perturbing branch 9999
        #z.assign(1.1*self.base + 0.2*perturbations[-1]) # used for r=4.8
        #z.assign(1.001*self.base + 0.0001*perturbations[-1]) # used for r=1.8 up to 4.2
        #z.assign(1.05*self.base + 0.2*perturbations[-1]) # used for r=2.88
        z.assign(1.1*self.base + 0.4*perturbations[-1]) #used for r=4.0
        # check the energy of the perturbed solution
        print("The energy of the perturbed solution: %s" % (assemble(problem.energy(z, params))/params[2]))

        # Just in case the problem does something
        # important in transform_guess
        task = ContinuationTask(taskid=0, oldparams=None, newparams=params,
                                freeindex=None, branchid=branchid, direction=+1)
        problem.transform_guess(z, task, self.io)

        self.bcs = problem.boundary_conditions(V, params)

        self.J = problem.energy(z, params)
        # Ugh. Problem-specific hacks here.
        h = avg(CellDiameter(mesh))
        s = FacetNormal(mesh)
        (u, _) = split(z)
        self.J = self.J + h**(-3)/2 * inner(jump(grad(u),s), jump(grad(u),s))*dS
        # End problem specific hacks

        #print("Initial energy: ", assemble(self.J))
        self.dJ = derivative(self.J, z, v)

        self.g = Function(V)
        self._assemble_residual = create_assembly_callable(self.dJ, tensor=self.g, bcs=self.bcs)

        self.HJ = derivative(self.dJ, z, w)
        self.H = allocate_matrix(self.HJ, bcs=self.bcs)
        self._assemble_hessian = create_assembly_callable(self.HJ, tensor=self.H, bcs=self.bcs)

        # Assemble the Riesz map
#        zero = Function(V)
#        sqnorm = problem.squared_norm(z, zero, params)
#        R = derivative(derivative(0.5 * sqnorm, z, v), z, w)
#        RMat = assemble(R, mat_type="aij").M.handle
#        riesz_ksp = PETSc.KSP.create(comm)
#        riesz_ksp.setOperators(Rmat)
#        riesz_ksp.setType("preonly")
#        riesz_ksp.pc.setType("cholesky")
#        riesz_ksp.pc.setFactorSolverType("mumps")
#        riesz_ksp.setOptionsPrefix("riesz_")
#        riesz_ksp.setFromOptions()
#        riesz_ksp.setUp()
#        self.riesz_ksp = riesz_ksp

        # Set up TAO solver
        tao = PETSc.TAO().create(comm)
        tao.setFromOptions()
        tao.setObjective(self.formObjective)
        tao.setGradient(self.formGradient)
        tao.setHessian(self.formHessian, self.H.petscmat)
        self.tao = tao

    def compute_stability(self, solution, problem):
        Z = solution.function_space()
        bcs = problem.boundary_conditions(Z, self.params)

        # first, we compute the stability
        trial = TrialFunction(Z)
        test = TestFunction(Z)
        F = problem.residual(solution, self.params, test)
        J = derivative(F, solution, trial)
        A = assemble(J, bcs=bcs, mat_type="aij")
        A = A.M.handle

        pc = PETSc.PC().create(backend.comm_world)
        pc.setOperators(A)
        pc.setType("cholesky")
        pc.setFactorSolverType("mumps")
        pc.setUp()
        F = pc.getFactorMatrix()
        (neg, zero, pos) = F.getInertia()
        print("stability %s: (-:%s, 0:%s, +:%s)" % (self.params, neg, zero, pos))
        stable = (neg, zero, pos)

        # if unstable, we then compute the eigenfunctions corresponding to negative eigenvalues
        if neg != 0:
            print("start computing eigenvalues for unstable eigenvalues...")
            M = inner(test, trial)*dx
            M = assemble(M, bcs=bcs, mat_type="aij")
            M = M.M.handle
            # Zero the rows and columns of M associated with bcs:
            lgmap = Z.dof_dset.lgmap
            for bc in bcs:
                M.zeroRowsColumns(lgmap.apply(bcdofs(bc)), diag=0)

            solver_parameters = {
                                "eps_gen_hermitian": None,
                                "eps_type": "krylovschur",
                                "eps_monitor_conv": None,
                                "eps_smallest_magnitude": None,
                                "eps_target": -8, #we can vary this target
                                "st_type": "sinvert",
                                }
            opts = PETSc.Options()
            for (key, val) in solver_parameters.items():
                opts[key] = val

            # Create the SLEPc eigensolver
            eps = SLEPc.EPS().create(comm=backend.comm_world)
            eps.setOperators(A, M)
            eps.setFromOptions()
            eps.solve()

            eigenvalues = []
            eigenfunctions = []
            for i in range(eps.getConverged()):
                eigenvalue = eps.getEigenvalue(i)
                assert eigenvalue.imag == 0
                eigenvalues.append(eigenvalue.real)
                eigenfunction = Function(Z, name="Eigenfunction")
                with eigenfunction.dat.vec_wo as vec:
                    eps.getEigenvector(i, vec)
                eigenfunctions.append(eigenfunction.copy(deepcopy=True))
            # save eigenfunctions
            for (eigenvalue, eigenfunction) in zip(eigenvalues, eigenfunctions):
                print("Got eigenvalue %s" % eigenvalue)
                filename = "output/eigenfunctions-aspectratio-%s/eigenvalue-%s.pvd" % (self.params[2], eigenvalue)
                pvd = File(filename, comm=backend.comm_world)
                problem.save_pvd(eigenfunction, pvd, self.params)
                print("Saved eigenfunction to %s." % filename)
            return (stable, eigenfunctions)
        else:
            print("The solution is stable!")
            return ((0,0,46743), [])

    def compute_functionals(self, sol, params):
        q = params[0]
        r = params[2]
        B = Constant(1e-5)
        a = Constant(-5*2)
        b = Constant(0)
        c = Constant(5*2)
        K = Constant(0.3)
        l = Constant(30)
        (u, d) = split(sol)
        Q = as_tensor([[d[0], d[1]],
                       [d[1], -d[0]]])
        I = as_matrix([[1,0],[0,1]])
        mat = grad(grad(u)) + q**2 * (Q+I/2) * u
        energy = assemble(self.problem.energy(sol, params)) /r
        indefEnergy = assemble(
            + a/2 * u**2 * dx
            + b/3 * u**3 * dx
            + c/4 * u**4 * dx
        )/r
        fourthEnergy = assemble(B*inner(mat,mat)*dx)/r
        elasticEnergy = assemble(K/2 * inner(grad(Q), grad(Q)) *dx)/r
        bulkEnergy = assemble(-l * tr(Q*Q) *dx + l * dot(tr(Q*Q), tr(Q*Q))*dx)/r
        sqL2 = assemble(u*u*dx)

        return [
                energy, indefEnergy, fourthEnergy,
                elasticEnergy, bulkEnergy, sqL2
               ]

    def solve(self):
        with self.z.dat.vec as z_:
            self.tao.solve(z_)

        # check the stability of the tao solution
        print("start checking the stability of the tao solution...")
        (stable, eigenfuncs) = self.compute_stability(self.z, self.problem)

        # compute the functionals of the tao-solved solution and save it in defcon info
        funcs = self.compute_functionals(self.z, self.params)
        self.io.save_solution(self.z, funcs, self.params, 9999) # 9999 as branchid for tao-solution
        self.io.save_stability(stable, [], [], self.params, 9999)

        # compare the difference between the original and tao-solved solutions
        difference = Function(self.z.function_space()).assign(self.z - self.base)
        pvd_diff = File("output/difference-aspectratio-%s/solution.pvd" % self.params[2], comm=MPI.COMM_WORLD)
        problem.save_pvd(difference, pvd_diff, self.params)

        # save the solution by Tao solver
        filename = "output/pvd/ratio-%s/solution-9999.pvd" % self.params[2]
        pvd = File(filename, comm=MPI.COMM_WORLD)
        problem.save_pvd(self.z, pvd, self.params)
        print("Saved tao solution to %s" % filename)

        # compute the energy (excluding the C0 IP penalization term)
        r = self.params[2]
        W = self.params[1]
        (_, d) = split(self.z)
        Q = as_tensor([[d[0], d[1]],
                       [d[1], -d[0]]])
        Q_vertical = as_tensor([[-1/2, 0], [0, 1/2]])
        scale_energy = problem.energy(self.z,self.params) - W/2 * inner(Q-Q_vertical, Q-Q_vertical) * (ds(1)+ds(2))
        side_energy = assemble(W/2 * inner(Q-Q_vertical, Q-Q_vertical) * (ds(1)+ds(2)))
        j = assemble(scale_energy)/r+ side_energy
        print("Energy value of the tao-solved solution: %s" % j)


    def formObjective(self, tao, x):
        with self.z.dat.vec_wo as x_:
            x.copy(x_)
        [bc.apply(self.z) for bc in self.bcs]

        return assemble(self.J)


    def formGradient(self, tao, x, G):
        with self.z.dat.vec_wo as x_:
            x.copy(x_)
        [bc.apply(self.z) for bc in self.bcs]

        self._assemble_residual()

        with self.g.dat.vec_ro as g_:
            g_.copy(G)
        #self.riesz_ksp.solve(self.g, G)


    def formHessian(self, tao, x, H, HP):
        with self.z.dat.vec_wo as x_:
            x.copy(x_)
        [bc.apply(self.z) for bc in self.bcs]

        self._assemble_hessian()


if __name__ == "__main__":
    problem = SmecticProblem()
    #ratio parameters shown in video
    ratios = linspace(1.0, 5.0, 21)[15:16:1]
    #branches with lowest energy
    branches = [140, 256, 959, 463, 898, 1575, 1575, 1072, 1072, 256, 256, 898, 898, 562, 1317, 1317, 268, 9999, 537, 1176, 0][15:16:1]
    for (ratio, branchid) in zip(ratios, branches):
        parameters = (30, 10, ratio)
        solver = TaoOptimisation(problem, parameters, branchid)
        solver.solve()

    #parameters = (30, 10, 4.0)
    #branchid = 1317
    #parameters = (30, 10, 2.88)
    #branchid = 256
