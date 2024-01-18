import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets, decomposition
import clip

model, preprocess = clip.load("ViT-B/32", device='cpu')
model.eval()
text_array = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
text_emb = clip.tokenize(text_array)
text_features = model.encode_text(text_emb).detach().cpu().numpy() # 21, 512

X = text_features.T
y = text_array

print('X shape', X.shape)
print('labels:', y)

projection = manifold.TSNE(n_components=2, init='pca', random_state=501)
# projection = manifold.Isomap(n_components=2)
# projection = decomposition.PCA(n_components=2)

X_proj = projection.fit_transform(X)

print('X_proj shape', X_proj.shape)

x_min, x_max = X_proj.min(0), X_proj.max(0)
X_norm = (X_proj - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
plt.title(str(projection))
for i in range(len(y)):
    class_name = y[i]
    color = 'red'
    if class_name in text_array[-5:]:
        color = 'blue'
    plt.scatter([X_norm[i, 0]], [X_norm[i, 1]], color=color)
    plt.text(X_norm[i, 0], X_norm[i, 1], class_name, color=color,
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()

