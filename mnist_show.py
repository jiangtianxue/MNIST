import matplotlib.pyplot as plt
import numpy as np

# 对产生的loss值和accuracy值进行读取	
def text_read(filename):
	file = open(filename, 'r')	
	data = []
	while True:
		line = file.readline()
		if not line:
			break
		data.append(line.split())
	for i in range(len(data)):
		for j in range(len(data[i])):
			data[i][j] = float(data[i][j])
	file.close()
	return data

data = text_read('loss_accus.txt')
print(data[1])


plt.title('loss and accuracy')
plt.plot(np.arange(len(data[0])), data[0], label='train loss')
plt.plot(np.arange(len(data[1])), data[1], label='train acces')
plt.plot(np.arange(len(data[2])), data[2], label='val loss')
plt.plot(np.arange(len(data[3])), data[3], 'm', label='val acces')
plt.legend()
plt.show()