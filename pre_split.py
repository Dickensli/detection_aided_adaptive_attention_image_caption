file = "./data/annotation/results_20130124.token"
tr_file = "./data/annotation/results_20130124_tr.token"
val_file = "./data/annotation/results_20130124_val.token"
tr = open(tr_file, "w+")
val = open(val_file, "w+")
tr_num = 110000

with open(file) as inf:
    c = 0
    line = inf.readline()

    while line != "":
        c += 1
        if c <= tr_num:
            tr.write(line)
        else:
            val.write(line)
        line = inf.readline()
    tr.close()
    val.close()

