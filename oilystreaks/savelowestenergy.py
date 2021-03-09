from firedrake import *
from defcon import *
import defcon.backend as backend
from oilystreaks import SmecticProblem
import os

problem = SmecticProblem()
io = problem.io()

parameters  = problem.parameters()
functionals = problem.functionals()
mesh = problem.mesh(backend.comm_world)
Z = problem.function_space(mesh)

io.setup(parameters, functionals, Z)

#params = __import__("oilystreaks").params
params = linspace(1.0, 5.0, 21)

filename = os.path.join("output/", "lowestenergy", "lowestenergy.pvd")
pvd = File(filename)
lowest_energy_branches_stabilities = []
for param in params:
    value = (30, 10, param)
    branches = io.known_branches(value)
    print("Known branches at %s: %s" % (value, branches))

    energy_branch = []
    for branch in branches:
        functionals = io.fetch_functionals([value], branch)
        energy = functionals[0][0]
        # save (energy, branch) in the list
        energy_branch.append(([energy], [branch]))
    # pick the lowest energy solution
    (energyvalue, branchid) = sorted(energy_branch, key=lambda x: x[0])[0]
    print("The lowest energy branch at %s: %s" % (value, branchid))
    stability = io.fetch_stability(value, branchid)
    lowest_energy_branches_stabilities.append((branchid, stability))
    sols = io.fetch_solutions(value, branchid)
    for sol in sols:
        problem.save_pvd(sol, pvd, value)

    #print("Save the lowest energy branch to " + filename)

with open('lowest_energy_branches_stabilities.txt', 'w') as output:
    for row in lowest_energy_branches_stabilities:
        output.write(str(row) + '\n')
