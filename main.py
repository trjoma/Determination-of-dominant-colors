import numpy as np
import cv2
from matplotlib import pyplot as plt

K = 5  #number of colors

img = cv2.imread('b2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret, label, center = cv2.kmeans(Z, K, None, criteria, 10,
                                cv2.KMEANS_RANDOM_CENTERS)

# Now separate the data, Note the flatten()
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

clr = []
for k in range(len(center)):
    clr.append("#{:02x}{:02x}{:02x}".format(center[k][0], center[k][1],
                                            center[k][2]))

print('Used colors:')
for c in range(len(clr)):
    print(f' {clr[c]}')

inp = list(np.unique(list(label), return_counts=True))

outp = []
bars = []
height = []
for i in range(K):
    bars.append(inp[0][i])
    height.append(inp[1][i])

y_pos = np.arange(len(bars))

plt.subplot(121), plt.imshow(res2)  # 'img' or 'res2'
plt.title('Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.pie(
    height, labels=clr, colors=clr, autopct='%.1f%%', startangle=90)
plt.title('Dominant colors'), plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
#cv2.destroyAllWindows()
