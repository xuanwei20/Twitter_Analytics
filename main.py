import csv
import numpy as np
import matplotlib.pyplot as plt

disaster_words = []
relevant_words = []
decisions = []

with open('main.csv', 'r') as dataset:
    lines = csv.reader(dataset)
    headings = next(lines)
    for row in lines:
        disaster_words.append(int(row[2]))
        relevant_words.append(int(row[3]))
        decisions.append(int(row[4]))

print(disaster_words)
print(relevant_words)
print(decisions)

plt.scatter(disaster_words, decisions, marker='*', color='blue')
plt.xlabel('disaster word number')
plt.ylabel('disaster')
plt.title('logistic regression', fontsize = 15)
plt.show()

plt.scatter(relevant_words, decisions, marker='*', color='purple')
plt.xlabel('relevant word number')
plt.ylabel('disaster')
plt.title('logistic regression', fontsize = 15)
plt.show()

id_variable = np.column_stack((disaster_words, relevant_words))
print(id_variable)
dv_variable = np.array(decisions).reshape(len(decisions),1)
print(dv_variable)
