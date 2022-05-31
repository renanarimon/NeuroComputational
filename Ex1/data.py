import csv

# open the file in the write mode
import random
if __name__ == '__main__':

    f = open(r'/', 'w')

    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    for i in range(1000):
        dataSet = []
        m = random.randint(-10000, 10000)
        n = random.randint(-10000, 10000)

        dataSet[0] = m / 100
        dataSet[1] = n / 100
        dataSet[2] = 1 if n / 100 > 1 else -1

        writer.writerow(dataSet)

    # close the file
    f.close()
