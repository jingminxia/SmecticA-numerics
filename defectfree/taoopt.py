from firedrake import *
from firedrake.petsc import PETSc
from firedrake.assemble import create_assembly_callable, allocate_matrix
from firedrake.preconditioners.patch import bcdofs
from slepc4py import SLEPc

from defcon import *
import defcon.backend as backend
from defectfree import SmecticProblem

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

        #z.assign(1.0*self.base + 0.08*perturbations[0]) # used for theta=1.201
        #z.assign(1.0*self.base + 0.04*perturbations[0]) #used for theta=1.405; get branch 9999 for theta=1.4137; get branch 10002 for theta=1.4137
        #z.assign(1.0*self.base + 0.004*perturbations[0]) #used for theta=1.4137 to get branch 10001 
        #z.assign(1.0*self.base + 0.1*perturbations[0]) #used for theta=1.4137 to get branch 10000
        #z.assign(1.0*self.base + 0.052*perturbations[1]) #used for theta=1.4137 to get the final stable solution
        #z.assign(1.0*self.base + 0.031*perturbations[1]) #used for theta=1.492 to get branch 9999
        #z.assign(1.000001*self.base + 0.09*perturbations[1]) #used for theta=1.492 to get branch 10000 (the stable one)
        #z.assign(1.0*self.base + 0.04*perturbations[0]) #used for theta=1.2566 to get branch 9999
        #z.assign(1.000001*self.base + 0.1*perturbations[1]) #used for theta=1.2566 to get branch 10000, 10001
        #z.assign(1.000001*self.base + 0.01*perturbations[0]) #used for theta=1.2566 to get branch 10002 (the stable one)
        #z.assign(1.001*self.base + 0.1*perturbations[0]) #used for theta=1.099
        z.assign(1.00001*self.base + 0.1*perturbations[0]) #used for theta=1.178
        #z.assign(1.0001*self.base - 0.06*perturbations[0]) #used for theta=1.507; does not work
        # check the energy of the perturbed solution
        print("The energy of the perturbed solution: %s" % (assemble(problem.energy(z, params))/params[1]))

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

        # Set up TAO solver
        tao = PETSc.TAO().create(comm)
        tao.setFromOptions()
        tao.setObjective(self.formObjective)
        tao.setGradient(self.formGradient)
        tao.setHessian(self.formHessian, self.H.petscmat)
        self.tao = tao

    def compute_stability(self, solution, problem):
        Z = solution.function_space()
        # have to explicitly include BCs here since it depends on params
        theta = self.params[2]
        bc1 = DirichletBC(Z.sub(1), as_vector([cos(theta)**2-1/2, -sin(theta)*cos(theta)]), 1)#bottom
        bc2 = DirichletBC(Z.sub(1), as_vector([cos(theta)**2-1/2, sin(theta)*cos(theta)]), 2)#top
        bcs = [bc1, bc2]

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
                filename = "output/eigenfunctions-theta-%s/eigenvalue-%s.pvd" % (self.params[2], eigenvalue)
                pvd = File(filename, comm=backend.comm_world)
                problem.save_pvd(eigenfunction, pvd, self.params)
                print("Saved eigenfunction to %s." % filename)
            return (stable, eigenfunctions)
        else:
            print("The solution is stable!")
            return ((0,0,46530), [])

    def compute_functionals(self, sol, params):
        q = params[0]
        r = params[1]
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
        pvd_diff = File("output/difference-theta-%s/solution.pvd" % self.params[2], comm=MPI.COMM_WORLD)
        problem.save_pvd(difference, pvd_diff, self.params)

        # save the solution by Tao solver
        filename = "output/pvd/theta-%s/solution-9999.pvd" % self.params[2]
        pvd = File(filename, comm=MPI.COMM_WORLD)
        problem.save_pvd(self.z, pvd, self.params)
        print("Saved tao solution to %s" % filename)

        # compute the energy (excluding the C0 IP penalization term)
        print("Energy value of the tao-solved solution: %s" % (assemble(problem.energy(self.z, self.params))/self.params[1]))

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
    # theta values shown in video
    thetas = linspace(0,pi/2,201)[0::10]
    # branches with lowest energy
    # only the last 7 branches are unstable
    branches = [345, 261, 261, 345, 261, 345, 336, 316, 258, 258, 186, 276, 186, 303, 12, 135, 10001, 303, 10002, 9999, 310][15:16:1]
    for (theta, branchid) in zip(thetas[15:16:1], branches):
        parameters = (30, 4.0, theta)
        solver = TaoOptimisation(problem, parameters, branchid)
        solver.solve()
    #parameters = (30, 4.0, 1.570796326794897)
    #branchid = 310
    #parameters = (30, 4.0, 1.201659189998096)
    #branchid = 135
    #parameters = (30, 4.0, 1.4058627124814325)
    #branchid = 269
