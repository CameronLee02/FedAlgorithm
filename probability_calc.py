import itertools
import matplotlib.pyplot as plt
import numpy as np

#used for brute force searching each position with 'm' malicious nodes being constant
def iterateStringsSingle(n,m):
    symbols = ['o'] * (n-m) + ['x'] * m
    unique_combinations = set(itertools.permutations(symbols))

    combinations = {}
    for string in unique_combinations:
        combinations[string] = 0
        if string[0] == 'o' and string[1] == 'x':
            combinations[string] += 1
        if string[-1] == 'o' and string[-2] == 'x':
            combinations[string] += 1
        for i in range(1, len(string) - 1):
            if string[i] == 'o'and string[i - 1] == 'x' and string[i + 1] == 'x':
                combinations[string] += 1    

    grouped_combinations = {}
    for key, value in combinations.items():
        if value not in grouped_combinations:
            grouped_combinations[value] = []
        grouped_combinations[value].append(''.join(key))   

    try:
        succesful_attacks = len(unique_combinations)-len(grouped_combinations[0])
    except:
        succesful_attacks = len(unique_combinations)

    print(f"Successful Attacks positions for n={n} m={m}: {succesful_attacks}.... Total positions available: {len(unique_combinations)}")
    print(f"Percentage of any node being attacked is: {round(succesful_attacks/len(unique_combinations) *100,2)}")

    for value in grouped_combinations.values():
        print(value)

#used for brute force searching each position with 'm' malicious nodes increasing
def iterateStrings(n):
    for j in range(1, n):
        symbols = ['x'] * (n-j) + ['o'] * j
        unique_combinations = set(itertools.permutations(symbols))

        combinations = {}
        for string in unique_combinations:
            combinations[string] = 0
            if string[0] == 'x' and string[1] == 'o':
                combinations[string] += 1
            if string[-1] == 'x' and string[-2] == 'o':
                combinations[string] += 1
            for i in range(1, len(string) - 1):
                if string[i] == 'x'and string[i - 1] == 'o' and string[i + 1] == 'o':
                    combinations[string] += 1    

        grouped_combinations = {}
        for key, value in combinations.items():
            if value not in grouped_combinations:
                grouped_combinations[value] = []
            grouped_combinations[value].append(''.join(key))   

        try:
            succesful_attacks = len(unique_combinations)-len(grouped_combinations[0])
        except:
            succesful_attacks = len(unique_combinations)

        print(f"Successful Attacks positions for n={n} m={j}: {succesful_attacks}.... Total positions available: {len(unique_combinations)}")
        print(f"Percentage of any node being attacked is: {round(succesful_attacks/len(unique_combinations) *100,2)}")

        count = 0
        for value in combinations.values():
            if value > 1:
                count += value-1
        print(count)

        total = 0
        for value, keys in grouped_combinations.items():
            try:
                total += (value/(n-j)) * len(keys)
            except:
                total += 0
        print(f"Percentage of a single targeted node being attacked is: {round(total/len(unique_combinations) *100,2)}")
        print('________________________________')
        #print(f"Non Successful Attacks positions for n={n} m={j}: {len(nonsuccessfulAttacks)}")


#equation used to calculate the probability of a single node being targeted with a sandwich attack
def singleNodePercentageEquation(n,m):
    return (((1/n)*(m/(n-1)))*2 + (m/n)*((m-1)/(n-1)))*100

#graphs the prob of a single node bing targeted in 'n' sample size with incrementing 'm' malicious nodes
def graphSingleNodePercentageRange(n):
    values = []
    for i in range(1,n):
        values.append(singleNodePercentageEquation(n,i))
    maliciousNodes = np.arange(1,n)
    plt.plot(maliciousNodes, values, marker = 'o')
    plt.xlabel("Number of Malicious Nodes in Sample Size")
    plt.ylabel("Percentage of an Attack Occuring")
    plt.xticks(maliciousNodes)
    plt.title(f"Percentage of an Attack Occuring on a Sample Size of {n} Based on the Number of Malicious Nodes")
    plt.show()

#graphs the prob of a single node bing targeted in a range of 'n' sample sizes with incrementing 'm' malicious nodes
def graphMultipleNodePercentageRange(nMin, nMax):
    legendvalues = []
    for j in range(nMin,nMax+1):
        values = []
        for i in range(1,j):
            values.append(singleNodePercentageEquation(j,i))
        maliciousNodes = np.arange(1,j)
        legendvalues.append(f"Sample size: {j}")
        plt.plot(maliciousNodes, values, marker = 'o')
    plt.xlabel("Number of Malicious Nodes")
    plt.ylabel("Percentage of an Attack Occuring")
    plt.xticks(maliciousNodes)
    plt.title(f"Percentage of an Attack Occuring on a Sample Sizes {nMin} to {nMax} Based on the Number of Malicious Nodes")
    plt.plot(maliciousNodes, np.full(nMax-1,5), color="black")
    legendvalues.append("5% Benchmark")
    plt.legend(legendvalues, loc="lower right")
    plt.show()

#graphs the prob of a single node bing targeted in a range of 'n' sample sizes with the same proportion 'm' malicious nodes
def graphMultipleNodePercentageSameM(nMin, nMax, m):
    yValues = []
    for i in range(nMin,nMax+1):
        yValues.append(singleNodePercentageEquation(i,m))
    xValues = np.arange(nMin, nMax+1)
    plt.plot(xValues, yValues, marker = 'o')
    plt.xlabel("Sample Size")
    plt.ylabel("Percentage of an Attack Occuring")
    plt.xticks(xValues)
    plt.title(f"Percentage of an Attack Occuring on Different Sample Sizes with {m} Malicious Nodes")
    plt.show()


#graphSingleNodePercentageRange(13)
#graphMultipleNodePercentageRange(10,30)
#graphMultipleNodePercentageSameM(10,50,3)
graphMultipleNodePercentageSameM(5,10,2)

#print(singleNodePercentageEquation(9,5))
#testMultipleIteration(6)
#print(singleNodePercentageEquation(6,3))
#iterateStringsSingle(6,3)