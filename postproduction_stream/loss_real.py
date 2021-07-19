import sys
import pandas as pd

log = sys.argv[1]

start = "TLaBeL|ml_iteration|1"
end = "TLaBeL|ml_iteration|-1"


def read_iteration(f, start, end, lines):
    start_flag = False
    while True:
        line = f.readline()
        if not line:
            return False
        if line.find(start) >= 0:
            start_flag = True
        if start_flag:
            lines.append(line)
        if line.find(end) >= 0:
            break
    return True


def trim_iteration(lines):
    n = 0
    i = 0
    for line in lines:
        if line.find("val_loss improved") >= 0:
            n = i
        i += 1
    lines = lines[1 : (n + 1)]
    return lines


with open(log) as f:
    train_loss = []
    val_loss = []
    while True:
        lines = []
        if not read_iteration(f, start, end, lines):
            break
        # print("After read_iteration, lines = \n", lines)
        lines = trim_iteration(lines)
        # print("After read_iteration, lines = \n", lines)
        lines = list(
            filter(lambda x: x.find("val_loss") >= 0 and x.find(" loss:") >= 0, lines)
        )
        # print("After filter, lines = \n", lines)
        for line in lines:
            tokens = line.split(" ")
            tl = float(tokens[7])
            vl = float(tokens[10])
            if vl > 1000:
                continue
            print(tl, vl)
            sys.stdout.flush()
            train_loss.append(tl)
            val_loss.append(vl)


df = pd.DataFrame(columns=["train_loss", "val_loss"])

df["train_loss"] = train_loss
df["val_loss"] = val_loss

df.to_csv("/tmp/losses.csv")
