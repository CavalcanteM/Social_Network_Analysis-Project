# Writes the obtained clusters on different files
def save_clusters(folder, c1, c2, c3, c4):
    with open(folder + "/cluster0.txt", "w") as f:
        for element in c1:
            f.write(element + "\n")

    with open(folder + "/cluster1.txt", "w") as f:
        for element in c2:
            f.write(element + "\n")

    with open(folder + "/cluster2.txt", "w") as f:
        for element in c3:
            f.write(element + "\n")

    with open(folder + "/cluster3.txt", "w") as f:
        for element in c4:
            f.write(element + "\n")