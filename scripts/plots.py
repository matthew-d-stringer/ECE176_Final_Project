import matplotlib.pyplot as plt

epochs = list(range(1, 29))
loss_values = [0.6512, 0.5633, 0.5518, 0.5383, 0.5402, 0.5345, 0.5274, 0.5277, 
               0.5280, 0.5248, 0.5304, 0.5390, 0.5421, 0.5372, 0.5264, 0.5227, 
               0.5111, 0.5134, 0.5088, 0.5132, 0.5104, 0.5086, 0.5110, 0.5082, 
               0.5112, 0.5092, 0.5093, 0.5085]

plt.figure(figsize=(10, 5))
plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b', label='Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)

plt.show()
