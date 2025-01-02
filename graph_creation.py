import matplotlib.pyplot as plt
import numpy as np
import csv


def cleanTimeData(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            data.append(line)
    headings = data[0]
    data_mean = np.zeros(shape=len(headings),dtype='float64')
    for row in data[1:]:
        for index in range(len(row)):
            data_mean[index] += float(row[index])
    
    number_of_entries = len(data)-1
    if number_of_entries > 0:
        data_mean /= number_of_entries 

    return data_mean

def cleanTransmissionData(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            data.append(line)
    #number of transmissions won't change over epoch as long as number of participates and number/size of paritions don't change
    return data[-1]

def graphDataTransmissions(file_names, tests, dataset):
    data_list = []
    for file_name in file_names:
        data= cleanTransmissionData(file_name)
        data_list.append(data)
    segments ={
        "PoW Procedure": [],
        "Route Generation": [],
        "Noise Calculation": [],
        "Model and Result Distribution": []
    }
    for data in data_list:
        segments["PoW Procedure"].append(int(data[0]))
        segments["Route Generation"].append(int(data[1]))
        segments["Noise Calculation"].append(int(data[2]))
        segments["Model and Result Distribution"].append(int(data[3]))
    
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(tests))

    for boolean, weight_count in segments.items():
        p = ax.bar(tests, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
    
    max_height = max(bottom)
    ax.set_ylim(0, max_height * 1.1)

    ax.set_title(f"Transmissions Used to Complete Each Prodecure in an Epoch on Dataset {dataset}")
    ax.set_ylabel("Number of Transmissions")
    ax.legend(loc="upper left", title="Procedures in an Epoch", bbox_to_anchor=(1, 1))
    plt.tight_layout() 
    plt.show()

#normal bar graph with everything on the same bar
def graphDataTime(file_names, tests, columns, dataset):
    data_list = []
    for file_name in file_names:
        data= cleanTimeData(file_name)
        data_list.append(data)

    segments = {key: [] for key in columns}

    for data in data_list:
        segments["PoW Procedure"].append(data[0])
        segments["Route Generation"].append(data[1])
        segments["Noise Calculation"].append(data[2])
        segments["Training"].append(data[3])
        segments["Key Generation"].append(data[4])
        segments["Encyrption"].append(data[5])
        segments["Decyrption"].append(data[6])
        segments["Aggregation"].append(data[7])
        segments["Model Updating"].append(data[8])
    
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(tests))

    for boolean, weight_count in segments.items():
        p = ax.bar(tests, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
    
    max_height = max(bottom)
    ax.set_ylim(0, max_height * 1.1)

    ax.set_title(f"Average Time Taken to Complete Each Procedure in an Epoch on Dataset {dataset}")
    ax.set_ylabel("Processing Time (s)")
    ax.legend(loc="upper left", title="Procedures in an Epoch", bbox_to_anchor=(1, 1))
    plt.tight_layout() 
    plt.show()

def graphCompareDataTime(file_name, dataset):
    data= cleanTimeData(file_name)
    server = {
        "Decyrption": data[6],
        "Key Generation": data[4],
        "Model Updating": data[8]
    }
    node = {
        "Encryption": data[5],
        "PoW Procedure": data[0],
        "Route Generation": data[1],
        "Noise Calculation": data[2],
        "Aggregation": data[7]
    }
    
    fig, ax = plt.subplots() 

    ax.set_title(f"Average Processing Time Taken by a Node and Server in an Epoch \nExcluding Training Time on Dataset {dataset}")
    ax.set_ylabel("Processing Time (s)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Node", "Server"])

    bottom_server = 0
    bottom_node = 0
    width = 0.5
    
    for boolean, weight_count in node.items():
        p = ax.bar(0, weight_count, width, label=boolean, bottom=bottom_node)
        bottom_node += weight_count
    
    for boolean, weight_count in server.items():
        p = ax.bar(1, weight_count, width, label=boolean, bottom=bottom_server)
        bottom_server += weight_count

    ax.legend(loc="upper left", title="Processing Steps", bbox_to_anchor=(1, 1))
    plt.tight_layout() 
    plt.show()
        
if __name__ == '__main__':
    acc_loss_scores_list = []
    file_names1 = ["results3/results3_times.csv"]
    file_names2 = ["results3/results3_transmissions.csv"]
    columns = ["Training", "PoW Procedure", "Route Generation", "Noise Calculation", "Key Generation", "Encyrption", "Decyrption", "Aggregation", "Model Updating"]
    tests = ("N = 6")
    graphDataTime(file_names1, tests, columns, "CIFAR10")
    graphDataTransmissions(file_names2, tests, "CIFAR10")
    graphCompareDataTime("results3/results3_times.csv", "CIFAR10")
