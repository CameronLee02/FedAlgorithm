import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

def cleanData(file_name):
    df = pd.read_csv(file_name)
    df_scores = df[['acc_score', 'loss_score']]
    df = df.drop(columns=['acc_score', 'loss_score'])
    mean_values = df.mean()
    data = mean_values.values.tolist()
    scores = df_scores.values.tolist()

    return data, scores

#normal bar graph with everything on the same bar
def graphDataTimeV1(data_list, tests):
    segments = {
        "Training": [],
        "Route Generation": [],
        "PoW Procedure": [],
        "Noise Calculation": []
    }
    values = []
    for data in data_list:
        values.append(data[::2])
    for data in values:
        segments["PoW Procedure"].append(data[0])
        segments["Route Generation"].append(data[1])
        segments["Noise Calculation"].append(data[2])
        segments["Training"].append(data[3])
    
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(tests))

    for boolean, weight_count in segments.items():
        p = ax.bar(tests, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title("Average Time Taken to Complete Each Procedure in an Epoch")
    ax.legend(loc="upper right")

    plt.show()

def graphDataTimeV2(data_list, tests):
    segments = {
        "Training": [],
        "Route Generation": [],
        "PoW Procedure": [],
        "Noise Calculation": []
    }
    values = []
    for data in data_list:
        values.append(data[::2])
    for data in values:
        segments["PoW Procedure"].append(data[0])
        segments["Route Generation"].append(data[1])
        segments["Noise Calculation"].append(data[2])
        segments["Training"].append(data[3])
    
    x = np.arange(len(tests))  
    width = 0.20 
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in segments.items():
        offset = width * multiplier 
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time (s)')
    ax.set_title('Average Time Taken to Complete Each Procedure in an Epoch')
    ax.set_xticks(x + width, tests)
    ax.set_yscale('log')
    ax.legend(loc='upper right', ncols=1)

    plt.show()

def graphDataTimeV3(data_list, tests):
    values = []
    for data in data_list:
        values.append(data[::2])
    mylabels = ["PoW Procedure", "Route Generation", "Noise Calculation", "Training"]
    # create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # plot each pie chart in a separate subplot
    ax1.pie(values[0], labels = mylabels)
    ax2.pie(values[1], labels = mylabels)

    plt.show()



        
if __name__ == '__main__':
    data_list = []
    acc_loss_scores_list = []
    file_names = ["results1", "results2"]
    for file_name in file_names:
        data, acc_loss_scores = cleanData(file_name)
        data_list.append(data)
        acc_loss_scores_list.append(acc_loss_scores)
    tests = ("N = 8","N = 6")
    graphDataTimeV3(data_list, tests)
