import glob

# import sys

fns = glob.glob("../../Outputs/305/aggregation_runs/stage0000/task00*/agg.bp")
fns.sort()

ondisk = glob.glob("*positions.npy")

theoretical = []

for i in range(8):
    for s in range(120):
        theoretical.append(f"{i}_{s}_positions.npy")

missing = set(theoretical) - set(ondisk)

print("missing = ", len(missing))

IS = {}

for m in missing:
    i, s, _ = m.split("_")

    a = int(s) // 12
    try:
        IS[a].append(int(i))
    except Exception as e:
        print(e)
        IS[a] = [int(i)]


print("IS = ", IS)

print("IS.keys() = ", IS.keys())

for k in IS.keys():
    print("k = ", k)
    print("len(v) = ", len(IS[k]))
