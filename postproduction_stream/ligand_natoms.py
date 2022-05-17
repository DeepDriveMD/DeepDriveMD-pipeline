import adios2

bpfile = "/p/gpfs1/yakushin/Outputs/306/aggregation_runs/stage0000/task0000/agg.bp/"

with adios2.open(bpfile, "r") as fh:
    #    print(dir(fh))
    steps = fh.steps()
    #    print(steps)
    ligands = list(fh.read("ligand", [], [], 0, steps))
    #    print(ligands)
    natoms = list(fh.read("natoms", [], [], 0, steps))


l_a = {}

for ligand, a in zip(ligands, natoms):
    if not (ligand in l_a):
        l_a[ligand] = []

    l_a[ligand].append(a)

print(l_a)


def compress(lll):
    L = lll.copy()
    L.append(-10000)

    ll = []
    prev = L[0]
    counter = 1
    for i in range(1, len(L)):
        if L[i] == prev:
            counter += 1
            prev = L[i]
        else:
            ll.append((prev, counter))
            prev = L[i]
            counter = 1

    return ll


print("=" * 20)

for i in l_a:
    l_a[i] = compress(l_a[i])

    print(i)
    print("\t", l_a[i])
