import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import csv



def cleanScoreData(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            data.append(line)
    headings = data[0]
    acc = np.zeros(shape=20,dtype='float64')
    loss = np.zeros(shape=20,dtype='float64')
    for index, row in enumerate(data[1:]):
        acc[index] = float(row[0])
        loss[index] = float(row[1])

    return acc, loss

def cleanTimeData(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            data.append(line)
    headings = data[0]
    epochTime = np.zeros(shape=20,dtype='float64')
    for index, row in enumerate(data[1:]):
        epochTime[index] = float(row[9])
    
    return epochTime

def cleanSizeData(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            data.append(line)
    headings = data[0]
    before_encryption = data[1][0]
    before_decryption = data[1][1]
    return before_encryption, before_decryption

def cleanTimeDataMean(file_name):
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

def cleanTransmissionsData(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            data.append(line)
    return data[1]
#normal bar graph with everything on the same bar
def graphDataTime(file_names, columns):
    data_mean_dict = {}
    for key in file_names:
        file = file_names[key]
        data_mean= cleanTimeDataMean(file)
        data_mean_dict[key] = data_mean

    xaxis = file_names.keys()
    yaxis_dict = {key: [] for key in columns}

    for key in data_mean_dict.keys():
        yaxis_dict["Noise Calculation"].append(data_mean_dict[key][0])
        yaxis_dict["Training"].append(data_mean_dict[key][1])
        yaxis_dict["Key Generation"].append(data_mean_dict[key][2])
        yaxis_dict["Encyrption"].append(data_mean_dict[key][3])
        yaxis_dict["Decyrption"].append(data_mean_dict[key][4])
        yaxis_dict["Aggregation"].append(data_mean_dict[key][5])
        yaxis_dict["Model Updating"].append(data_mean_dict[key][6])
    
    print(yaxis_dict)
    total_runtime = {
        "SVHN": 0,
        "MNIST": 0,
        "CIFAR-10": 0
    }
    for key in yaxis_dict:
        data = yaxis_dict[key]
        total_runtime["SVHN"] += data[0]
        total_runtime["MNIST"] += data[1]
        total_runtime["CIFAR-10"] += data[2]
    SVHNRuntime = total_runtime["SVHN"]
    SVHNTrainRuntime = yaxis_dict["Training"][0]
    MNISTRuntime = total_runtime["MNIST"]
    MNISTTrainRuntime = yaxis_dict["Training"][1]
    CIFA10Runtime = total_runtime["CIFAR-10"]
    CIFA10TrainRuntime = yaxis_dict["Training"][2]
    print(f"Total Runtime SVHN: {SVHNRuntime}.... Runtime of Training: {SVHNTrainRuntime}...." +
          f"Percentage of our scheme runtime: {(SVHNTrainRuntime/SVHNRuntime)*100}")
    print(f"Total Runtime MNIST: {MNISTRuntime}.... Runtime of Training: {MNISTTrainRuntime}...." +
          f"Percentage of our scheme runtime: {(MNISTTrainRuntime/MNISTRuntime)*100}")
    print(f"Total Runtime CIFAR-10: {CIFA10Runtime}.... Runtime of Training: {CIFA10TrainRuntime}...." +
          f"Percentage of our scheme runtime: {(CIFA10TrainRuntime/CIFA10Runtime)*100}")
    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(xaxis))

    for boolean, weight_count in yaxis_dict.items():
        p = ax.bar(xaxis, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
    
    max_height = max(bottom)
    ax.set_ylim(0, max_height * 1.1)

    ax.set_ylabel("Time (seconds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout() 
    plt.show()

def graphCompareDataTime(file_name):
    data= cleanTimeDataMean(file_name)
    server = {
        "Decyrption": data[4],
        "Key Generation": data[2],
        "Model Updating": data[6]
    }
    node = {
        "Encryption": data[3],
        "Noise Calculation": data[0],
        "Aggregation": data[5]
    }
    
    fig, ax = plt.subplots(figsize=(10, 6)) 

    ax.set_ylabel("Time (seconds)")
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

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout() 
    plt.show()

def graphCompareDatasetScoresBaseFL(file_names):
    xaxis = np.arange(1,21)
    yaxis_dict = {}
    for key in file_names:
        file = file_names[key]
        acc, loss = cleanScoreData(file)
        yaxis_dict[key] = [acc, loss]
    
    fig, ax = plt.subplots(figsize=(10, 6)) 

    for key in yaxis_dict:
        yaxis = yaxis_dict[key][0]
        if "FedAVG" in key:
            ax.plot(xaxis, yaxis, marker='o', linestyle='--', label=key)
        else:
            ax.plot(xaxis, yaxis, marker='o', linestyle='-', label=key)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')

    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout() 
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6)) 

    for key in yaxis_dict:
        yaxis = yaxis_dict[key][1]
        if "FedAVG" in key:
            ax.plot(xaxis, yaxis, marker='o', linestyle='--', label=key)
        else:
            ax.plot(xaxis, yaxis, marker='o', linestyle='-', label=key)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout() 
    plt.show()

def graphCompareDatasetEpochTime(file_names):
    xaxis = np.arange(1,21)
    yaxis_dict = {}
    for key in file_names:
        file = file_names[key]
        data_mean= cleanTimeData(file)
        yaxis_dict[key] = data_mean

    fig, ax = plt.subplots(figsize=(10, 6)) 

    for key in yaxis_dict:
        yaxis = yaxis_dict[key]
        ax.plot(xaxis, yaxis, marker='o', linestyle='--', label=key)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')

    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout() 
    plt.show()

def graphEncyptDecryptSizeVTime(file_names):
    xyaxis_dict = {
        "Before Decryption" : [[],[]],
        "Before Encryption" : [[],[]]
    }
    for key in file_names:
        file = file_names[key]
        before_encryption_size, before_decryption_size = cleanSizeData(file[1])
        data_mean = cleanTimeDataMean(file[0])
        before_encryption_time = data_mean[3]
        before_decryption_time = data_mean[4]

        xyaxis_dict["Before Decryption"][0].append(float(before_decryption_size))
        xyaxis_dict["Before Decryption"][1].append(float(before_decryption_time))

        xyaxis_dict["Before Encryption"][0].append(float(before_encryption_size))
        xyaxis_dict["Before Encryption"][1].append(float(before_encryption_time))

    fig, ax = plt.subplots(figsize=(10, 6))
    yaxis = xyaxis_dict["Before Encryption"][1]
    xaxis = xyaxis_dict["Before Encryption"][0]
    ax.scatter(xaxis, yaxis, color='orange')  
    ax.plot(xaxis, yaxis, linestyle='--', color='orange', alpha=0.6, label="Before Encryption")
    ax.set_xlabel('Size (bytes)')
    ax.set_ylabel('Time (seconds)')

    for i, (x, y, label) in enumerate(zip(xaxis, yaxis, file_names.keys())):
        ax.annotate(f'{label}', xy=(x, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)

    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='x')
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout() 
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    yaxis = xyaxis_dict["Before Decryption"][1]
    xaxis = xyaxis_dict["Before Decryption"][0]
    ax.scatter(xaxis, yaxis, color='blue') 
    ax.plot(xaxis, yaxis, linestyle='--', color='blue', alpha=0.6, label="Before Decrytpion")
    ax.set_xlabel('Size (bytes)')
    ax.set_ylabel('Time (seconds)')

    for i, (x, y, label) in enumerate(zip(xaxis, yaxis, file_names.keys())):
        ax.annotate(f'{label}', xy=(x, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)

    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='x')
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout() 
    plt.show()

def graphCompareOverhead(file_names, columns, xlabel):
    data_mean_dict = {}
    for key in file_names:
        file = file_names[key]
        data_mean= cleanTimeDataMean(file)
        data_mean_dict[key] = data_mean

    xaxis = file_names.keys()
    yaxis_dict = {key: [] for key in columns}

    for key in data_mean_dict:
        yaxis_dict["Noise Calculation"].append(data_mean_dict[key][0])
        yaxis_dict["Key Generation"].append(data_mean_dict[key][2])
        yaxis_dict["Encyrption"].append(data_mean_dict[key][3])
        yaxis_dict["Decyrption"].append(data_mean_dict[key][4])
        yaxis_dict["Aggregation"].append(data_mean_dict[key][5])
        yaxis_dict["Model Updating"].append(data_mean_dict[key][6])
    
    fig, ax = plt.subplots(figsize=(10, 6)) 

    for key in yaxis_dict:
        yaxis = yaxis_dict[key]
        ax.plot(xaxis, yaxis, marker='o', linestyle='--', label=key)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Time (seconds)')

    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout() 
    plt.show()


def graphCompareOverheadClient(file_names, columns, xlabel):
    data_mean_dict = {}
    for key in file_names:
        file = file_names[key]
        data_mean= cleanTimeDataMean(file)
        data_mean_dict[key] = data_mean

    xaxis = file_names.keys()
    yaxis_dict = {key: [] for key in columns}

    for key in data_mean_dict:
        yaxis_dict["Noise Calculation"].append(data_mean_dict[key][0])
        yaxis_dict["Encyrption"].append(data_mean_dict[key][3])
        yaxis_dict["Aggregation"].append(data_mean_dict[key][5])
    
    fig, ax = plt.subplots(figsize=(10, 6)) 

    for key in yaxis_dict:
        yaxis = yaxis_dict[key]
        ax.plot(xaxis, yaxis, marker='o', linestyle='--', label=key)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Time (seconds)')

    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout() 
    plt.show()

def graphCompareOverheadServer(file_names, columns, xlabel):
    data_mean_dict = {}
    for key in file_names:
        file = file_names[key]
        data_mean= cleanTimeDataMean(file)
        data_mean_dict[key] = data_mean

    xaxis = file_names.keys()
    yaxis_dict = {key: [] for key in columns}

    for key in data_mean_dict:
        yaxis_dict["Decyrption"].append(data_mean_dict[key][4])
        yaxis_dict["Key Generation"].append(data_mean_dict[key][2])
        yaxis_dict["Model Updating"].append(data_mean_dict[key][6])
    
    fig, ax = plt.subplots(figsize=(10, 6)) 

    for key in yaxis_dict:
        yaxis = yaxis_dict[key]
        ax.plot(xaxis, yaxis, marker='o', linestyle='--', label=key)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Time (seconds)')

    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout() 
    plt.show()

def graphCompareTransmissionsOverPartitions(file_names, columns):
    transmissions_dict = {}
    for key in file_names:
        file = file_names[key]
        transmissions= cleanTransmissionsData(file)
        transmissions_dict[key] = transmissions
    
    xaxis = file_names.keys()
    yaxis_dict = {key: [] for key in columns}
    for key in transmissions_dict:
        yaxis_dict["Noise Calculation"].append(int(transmissions_dict[key][0]))
        yaxis_dict["Model and Weight Distribution"].append(int(transmissions_dict[key][1]))

    fig, ax = plt.subplots(figsize=(10, 6)) 

    for key in yaxis_dict:
        yaxis = yaxis_dict[key]
        ax.plot(xaxis, yaxis, marker='o', linestyle='--', label=key)
    ax.set_xlabel('Number of Partitioned Chains')
    ax.set_ylabel('Number of Transmissions')
    
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout() 
    plt.show()

        
if __name__ == '__main__':
    ### COMPARE SERVER AND CLIENT COMPUTATION OVERHEAD
    # THIS IS USED TO COMPARE THE AVG PROCEDURE TIMES (STACKED BAR CHART) FOR SERVER AND CLIENT, OF OUR SCHEME ON MNIST (EXCLUDING TRAINING TIME)

    graph_compare_client_server_overhead = "../Results2/MNIST_baseline/MNIST_baseline_times.csv"
    graphCompareDataTime(graph_compare_client_server_overhead)

    ### COMPARE TRANSMISSIONS OVER DIFFERENT PARTITIONS ON MNIST

    graph_compare_number_partition_files = {
        "1": "../Results2/MNIST_baseline/MNIST_baseline_transmissions.csv",
        "2": "../Results2/MNIST_baseline_partition_2/MNIST_baseline_partition_2_transmissions.csv",
        "3": "../Results2/MNIST_baseline_partition_3/MNIST_baseline_partition_3_transmissions.csv"
    }
    columns = ["Noise Calculation", "Model and Weight Distribution"]
    graphCompareTransmissionsOverPartitions(graph_compare_number_partition_files, columns)

#------------------------------------------------------------------------------------------------------------------------------#
    ### COMPARE DATASETS
    # THIS IS USED TO COMPARE THE ACCURACY AND LOSS OF OUR SCHEME VS FEDAVG SCHEME ACROSS DIFFERENT DATASETS
    graph_compare_dataset_scores_files = {
        "MNIST Our Scheme": "../Results2/MNIST_baseline/MNIST_baseline_scores.csv",
        "MNIST FedAVG Scheme": "../Results2/MNIST_baseline_standard_fl/MNIST_baseline_standard_fl_scores.csv",
        "CIFAR10 Our Scheme": "../Results2/CIFAR10_baseline/CIFAR10_baseline_scores.csv",
        "CIFAR10 FedAVG Scheme": "../Results2/CIFAR10_baseline_standard_fl/CIFAR10_baseline_standard_fl_scores.csv",
        "SVHN Our Scheme": "../Results2/SVHN_baseline/SVHN_baseline_scores.csv",
        "SVHN FedAVG Scheme": "../Results2/SVHN_baseline_standard_fl/SVHN_baseline_standard_fl_scores.csv"
    }
    graphCompareDatasetScoresBaseFL(graph_compare_dataset_scores_files)

    # THIS IS USED TO COMPARE THE OVERHEAD OF OUR SCHEME FOR CLIENTS ACROSS DIFFERENT DATASETS 
    '''
    graph_compare_dataset_time_files = {
        "SVHN": "../Results/SVHN_baseline/SVHN_baseline_times.csv",
        "MNIST": "../Results/MNIST_baseline/MNIST_baseline_times.csv",
        "CIFAR-10": "../Results/CIFAR10_baseline/CIFAR10_baseline_times.csv"
    }
    columns = ["PoW Procedure", "Route Generation", "Noise Calculation", "Encyrption", "Aggregation"]
    graphCompareOverheadClient(graph_compare_dataset_time_files, columns, 'Datasets')'''

    # THIS IS USED TO COMPARE THE OVERHEAD OF OUR SCHEME FOR SERVER ACROSS DIFFERENT DATASETS 
    '''
    graph_compare_dataset_time_files = {
        "SVHN": "../Results/SVHN_baseline/SVHN_baseline_times.csv",
        "MNIST": "../Results/MNIST_baseline/MNIST_baseline_times.csv",
        "CIFAR-10": "../Results/CIFAR10_baseline/CIFAR10_baseline_times.csv"
    }
    columns = ["Decyrption", "Key Generation", "Model Updating"]
    graphCompareOverheadServer(graph_compare_dataset_time_files, columns, 'Datasets')'''

    # THIS IS USED TO COMPARE THE OVERHEAD OF OUR SCHEME FOR BOTH CLIENT AND SERVER ACROSS DIFFERENT DATASETS 

    graph_compare_dataset_time_files = {
        "SVHN": "../Results2/SVHN_baseline/SVHN_baseline_times.csv",
        "MNIST": "../Results2/MNIST_baseline/MNIST_baseline_times.csv",
        "CIFAR-10": "../Results2/CIFAR10_baseline/CIFAR10_baseline_times.csv"
    }
    columns = ["Noise Calculation", "Encyrption", "Aggregation", "Decyrption", "Key Generation", "Model Updating"]
    graphCompareOverhead(graph_compare_dataset_time_files, columns, 'Datasets')

    # THIS IS USED TO COMPARE THE EPOCH TIMES OF OUR SCHEME ACROSS DIFFERENT DATASETS
    '''
    graph_compare_dataset_time_files = {
        "SVHN": "../Results/SVHN_baseline/SVHN_baseline_times.csv",
        "MNIST": "../Results/MNIST_baseline/MNIST_baseline_times.csv",
        "CIFAR-10": "../Results/CIFAR10_baseline/CIFAR10_baseline_times.csv"
    }
    graphCompareDatasetEpochTime(graph_compare_dataset_time_files)'''

    # THIS IS USED TO COMPARE THE AVG EPOCH TIMES (STACKED BAR CHART) OF OUR SCHEME ACROSS DIFFERENT DATASETS

    graph_compare_dataset_time_files = {
        "SVHN": "../Results2/SVHN_baseline/SVHN_baseline_times.csv",
        "MNIST": "../Results2/MNIST_baseline/MNIST_baseline_times.csv",
        "CIFAR-10": "../Results2/CIFAR10_baseline/CIFAR10_baseline_times.csv"
    }
    columns = ["Training", "Noise Calculation", "Key Generation", "Encyrption", "Decyrption", "Aggregation", "Model Updating"]
    graphDataTime(graph_compare_dataset_time_files, columns)

    # THIS IS USED TO COMPARE THE AVG EPOCH TIMES (STACKED BAR CHART AND EXCLUDING TRAINING) OF OUR SCHEME ACROSS DIFFERENT DATASETS
    '''
    graph_compare_dataset_time_files = {
        "SVHN": "../Results/SVHN_baseline/SVHN_baseline_times.csv",
        "MNIST": "../Results/MNIST_baseline/MNIST_baseline_times.csv",
        "CIFAR-10": "../Results/CIFAR10_baseline/CIFAR10_baseline_times.csv"
    }
    columns = ["PoW Procedure", "Route Generation", "Noise Calculation", "Key Generation", "Encyrption", "Decyrption", "Aggregation", "Model Updating"]
    graphDataTime(graph_compare_dataset_time_files, columns)'''

    # THIS IS USED TO COMPARE THE SIZE TO SIZE OF ENCRYPTION/DECRYPTION OF OUR SCHEME ACROSS DIFFERENT DATASETS
    
    graph_compare_dataset_time_files = {
        "SVHN": ["../Results2/SVHN_baseline/SVHN_baseline_times.csv", "../Results2/SVHN_baseline/SVHN_baseline_sizes.csv"],
        "MNIST": ["../Results2/MNIST_baseline/MNIST_baseline_times.csv", "../Results2/MNIST_baseline/MNIST_baseline_sizes.csv"],
        "CIFAR-10": ["../Results2/CIFAR10_baseline/CIFAR10_baseline_times.csv", "../Results2/CIFAR10_baseline/CIFAR10_baseline_sizes.csv"]
    }
    graphEncyptDecryptSizeVTime(graph_compare_dataset_time_files)
#------------------------------------------------------------------------------------------------------------------------------#

    ### CLIENTS
    # THIS IS USED TO COMPARE THE OVERHEAD OF OUR SCHEME FOR CLIENTS ACROSS A CHANGE IN NUMBER OF CLIENTS ON MNIST
    '''
    graph_compare_number_clients_files = {
        "8": "../Results/MNIST_baseline_nodes_8/MNIST_baseline_nodes_8_times.csv",
        "10": "../Results/MNIST_baseline/MNIST_baseline_times.csv",
        "12": "../Results/MNIST_baseline_nodes_12/MNIST_baseline_nodes_12_times.csv"
    }
    columns = ["PoW Procedure", "Route Generation", "Noise Calculation", "Encyrption", "Aggregation"]
    graphCompareOverheadClient(graph_compare_number_clients_files, columns, 'Number of Clients')'''

    # THIS IS USED TO COMPARE THE OVERHEAD OF OUR SCHEME FOR CLIENTS ACROSS A CHANGE IN NUMBER PARTITIONED CHAINS ON MNIST
    '''
    graph_compare_number_partition_files = {
        "1": "../Results/MNIST_baseline/MNIST_baseline_times.csv",
        "2": "../Results/MNIST_baseline_partition_2/MNIST_baseline_partition_2_times.csv",
        "3": "../Results/MNIST_baseline_partition_3/MNIST_baseline_partition_3_times.csv"
    }
    columns = ["PoW Procedure", "Route Generation", "Noise Calculation", "Encyrption", "Aggregation"]
    graphCompareOverheadClient(graph_compare_number_partition_files, columns, 'Number of Partitioned Chains')'''

    # THIS IS USED TO COMPARE THE OVERHEAD OF OUR SCHEME FOR CLIENTS ACROSS A CHANGE IN DIFFICULTY OF POW ON MNIST
    '''
    graph_compare_pow_difficulty_files = {
        "3": "../Results/MNIST_baseline_pow_3/MNIST_baseline_pow_3_times.csv",
        "4": "../Results/MNIST_baseline/MNIST_baseline_times.csv",
        "5": "../Results/MNIST_baseline_pow_5/MNIST_baseline_pow_5_times.csv"
    }
    columns = ["PoW Procedure", "Route Generation", "Noise Calculation", "Encyrption", "Aggregation"]
    graphCompareOverheadClient(graph_compare_pow_difficulty_files, columns, 'Difficulty of PoW')'''

#------------------------------------------------------------------------------------------------------------------------------#
    ### SERVER
    # THIS IS USED TO COMPARE THE OVERHEAD OF OUR SCHEME FOR THE SERVER ACROSS A CHANGE IN NUMBER OF CLIENTS ON MNIST
    '''
    graph_compare_number_clients_files = {
        "8": "../Results/MNIST_baseline_nodes_8/MNIST_baseline_nodes_8_times.csv",
        "10": "../Results/MNIST_baseline/MNIST_baseline_times.csv",
        "12": "../Results/MNIST_baseline_nodes_12/MNIST_baseline_nodes_12_times.csv"
    }
    columns = ["Decyrption", "Key Generation", "Model Updating"]
    graphCompareOverheadServer(graph_compare_number_clients_files, columns, 'Number of Clients')'''

    # THIS IS USED TO COMPARE THE OVERHEAD OF OUR SCHEME FOR THE SERVER ACROSS A CHANGE IN NUMBER PARTITIONED CHAINS ON MNIST
    '''
    graph_compare_number_partition_files = {
        "1": "../Results/MNIST_baseline/MNIST_baseline_times.csv",
        "2": "../Results/MNIST_baseline_partition_2/MNIST_baseline_partition_2_times.csv",
        "3": "../Results/MNIST_baseline_partition_3/MNIST_baseline_partition_3_times.csv"
    }
    columns = ["Decyrption", "Key Generation", "Model Updating"]
    graphCompareOverheadServer(graph_compare_number_partition_files, columns, 'Number of Partitioned Chains')'''

    # THIS IS USED TO COMPARE THE OVERHEAD OF OUR SCHEME FOR THE SERVER ACROSS A CHANGE IN DIFFICULTY OF POW ON MNIST
    '''
    graph_compare_pow_difficulty_files = {
        "3": "../Results/MNIST_baseline_pow_3/MNIST_baseline_pow_3_times.csv",
        "4": "../Results/MNIST_baseline/MNIST_baseline_times.csv",
        "5": "../Results/MNIST_baseline_pow_5/MNIST_baseline_pow_5_times.csv"
    }
    columns = ["Decyrption", "Key Generation", "Model Updating"]
    graphCompareOverheadServer(graph_compare_pow_difficulty_files, columns, 'Difficulty of PoW')'''


#------------------------------------------------------------------------------------------------------------------------------#
    ### SERVER & CLIENTS
    # THIS IS USED TO COMPARE THE OVERHEAD OF OUR SCHEME FOR THE SERVER & CLIENTS ACROSS A CHANGE IN NUMBER OF CLIENTS ON MNIST
    graph_compare_number_clients_files = {
        "10": "../Results2/MNIST_baseline/MNIST_baseline_times.csv",
        "15": "../Results2/MNIST_baseline_nodes_15/MNIST_baseline_nodes_15_times.csv",
        "20": "../Results2/MNIST_baseline_nodes_20/MNIST_baseline_nodes_20_times.csv"
    }
    columns = ["Noise Calculation","Key Generation", "Encyrption", "Decyrption", "Aggregation", "Model Updating"]
    graphCompareOverhead(graph_compare_number_clients_files, columns, 'Number of Clients')

    # THIS IS USED TO COMPARE THE OVERHEAD OF OUR SCHEME FOR THE SERVER & CLIENTS ACROSS A CHANGE IN NUMBER PARTITIONED CHAINS ON MNIST

    graph_compare_number_partition_files = {
        "1": "../Results2/MNIST_baseline/MNIST_baseline_times.csv",
        "2": "../Results2/MNIST_baseline_partition_2/MNIST_baseline_partition_2_times.csv",
        "3": "../Results2/MNIST_baseline_partition_3/MNIST_baseline_partition_3_times.csv"
    }
    columns = ["Noise Calculation", "Key Generation", "Encyrption", "Decyrption", "Aggregation", "Model Updating"]
    graphCompareOverhead(graph_compare_number_partition_files, columns, 'Number of Partitioned Chains')

    # THIS IS USED TO COMPARE THE OVERHEAD OF OUR SCHEME FOR THE SERVER & CLIENTS ACROSS A CHANGE IN DIFFICULTY OF POW ON MNIST
    '''
    graph_compare_pow_difficulty_files = {
        "3": "../Results/MNIST_baseline_pow_3/MNIST_baseline_pow_3_times.csv",
        "4": "../Results/MNIST_baseline/MNIST_baseline_times.csv",
        "5": "../Results/MNIST_baseline_pow_5/MNIST_baseline_pow_5_times.csv"
    }
    columns = ["PoW Procedure", "Route Generation", "Noise Calculation", "Key Generation", "Encyrption", "Decyrption", "Aggregation", "Model Updating"]
    graphCompareOverhead(graph_compare_pow_difficulty_files, columns, 'Difficulty of PoW')'''