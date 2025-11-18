import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

from core.mlp import MLP
from core.loss import bce


def get_data():
    from ucimlrepo import fetch_ucirepo 
    
    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 
    
    # data (as pandas dataframes) 
    X = mushroom.data.features 
    y = mushroom.data.targets 
    
    return X, y


X, y = get_data()

X['stalk-root'] = X['stalk-root'].fillna('-')

X = X.drop(['veil-type', 'gill-attachment', 'veil-color', 'stalk-color-below-ring'], axis=1)

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(X)

X = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(X.columns), index=X.index)


y['poisonous'] = y['poisonous'].map({'p': 0, 'e': 1})

X = X.to_numpy()

y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

nn = MLP(input_size=X.shape[1], hidden1=256, hidden2=64)

lr = 1e-2
epochs = 200
batch_size=32

loss_values = []

test_loss_values = []

with open('loss.txt', 'w') as f:
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            y_pred = nn.forward(X_batch)


            loss = bce(y_pred, y_batch)

            epoch_loss += loss * len(X_batch)

            nn.backward(y_batch, y_pred)

            nn.update(lr)

            nn.zero_grad()

        loss_values.append(epoch_loss / len(X_train))
        if (epoch+1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(X_train):.4f}")


print(f"X_test: {X_test.shape}")
y_pred = nn.forward(X_test)

y_pred_labels = (y_pred > 0.5).astype(int)
y_true = y_test


print(f"y_pred: {y_pred}")
print(f"y_pred_labels: {y_pred_labels}")
print(f"y_true: {y_true}")


accuracy = accuracy_score(y_true, y_pred_labels)
precision = precision_score(y_true, y_pred_labels)
recall = recall_score(y_true, y_pred_labels)
f1 = f1_score(y_true, y_pred_labels)
conf_matrix = confusion_matrix(y_true, y_pred_labels)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"confusion_matrix:\n{conf_matrix}")

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(range(epochs), loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

