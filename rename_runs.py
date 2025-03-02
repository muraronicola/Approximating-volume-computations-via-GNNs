import os


#For each file in the directory ./runs, it finds the number of the file and renames it to run + number

max = 100
indici = 1

for i in range(1, max):
    if os.path.exists("./runs/run" + str(i) + ".csv"):
        os.rename("./runs/run" + str(i) + ".csv", "./runs/run" + str(indici) + ".csv")
        os.rename("./runs/run" + str(i) + ".png", "./runs/run" + str(indici) + ".png")
        os.rename("./runs/run" + str(i) + ".txt", "./runs/run" + str(indici) + ".txt")
        os.rename("./runs/run" + str(i) + "_train_predictions.csv", "./runs/run" + str(indici) + "_train_predictions.csv")
        os.rename("./runs/run" + str(i) + "_dev_predictions.csv", "./runs/run" + str(indici) + "_dev_predictions.csv")
        os.rename("./runs/run" + str(i) + "_test_predictions.csv", "./runs/run" + str(indici) + "_test_predictions.csv")
        indici += 1
        
print("Done")