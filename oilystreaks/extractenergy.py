from firedrake import *
from defcon import *
import defcon.backend as backend
from oilystreaks import SmecticProblem

problem = SmecticProblem()
io = problem.io()

parameters  = problem.parameters()
functionals = problem.functionals()
mesh = problem.mesh(backend.comm_world)
Z = problem.function_space(mesh)

io.setup(parameters, functionals, Z)

energy_branch = []
params = [2.88, 1.5, 4.0]
for param in params:
    value = (30, 10, param)
    branches = io.known_branches(value)
    print("Known branches at %s: %s" % (value, branches))

    for branch in branches:
        functionals = io.fetch_functionals([value], branch)
        energy = functionals[0][0]
        # save (energy, branch) in the list
        energy_branch.append(([energy], [branch]))

    with open('energy-value-(30,10,%s).txt' % param, 'w') as output:
        for row in energy_branch:
            output.write(str(row) + '\n')
