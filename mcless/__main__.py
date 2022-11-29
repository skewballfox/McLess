# %%
import numpy as np
from numpy import float64, floating
from numpy.typing import NDArray

import seaborn as sbn
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from mcless.model import McLess

# %%

# data_frame=pl.DataFrame(load_iris(as_frame=True))
data_read = load_iris()
X = data_read.data
Y = data_read.target
data_file = data_read.filename
targets = data_read.target_names
features = data_read.feature_names
print(targets)
print(f"feature names: {features}")
print(f"X shape: {X.shape}, Y shape {Y.shape}")

# %% [markdown]
# # Settings

# %%
N, d = X.shape
labelset = set(Y)
number_of_classifications = len(labelset)
# number of iterations
run_count = 100
training_data_percent = 0.7e0
test_data_percent = 1 - training_data_percent

# %%
# split_time_arr=np.zeros(5)
# split_time_start=split_time_stop=time.time()
# for i in range(5):
#     split_time_start=time.time()
#     split_time_start=time.time()
#     split_time_arr

# %%
Accuracy: NDArray[float64] = np.zeros(run_count)
from sklearn.neighbors import KNeighborsClassifier

# classifier=KNeighborsClassifier(number_of_classifications)
classifier = McLess()
time_total_start = time.time()

time_train_start = time_train = avg_train_time = time.time()
for i in range(run_count):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X, Y, test_size=test_data_percent, random_state=i, stratify=Y
    )
    print(X.shape)
    time_train_start = time.time()
    classifier.fit(Xtrain, Ytrain)
    time_train = time.time() - time_train_start
    # update average training time and accuracy
    avg_train_time = (avg_train_time + time_train) / 2
    Accuracy[i] = classifier.score(Xtest, Ytest)
    # print(f"accuracy for loop {i}: {Accuracy[i]}")

time_total = time.time() - time_total_start

# %% [markdown]
# # Display Stats

# %%
print(f"For dataset: {data_file}")
print(
    f"Average Accuracy {Accuracy.mean()}, standard deviation {Accuracy.std()}, Average training time {avg_train_time}"
)
print(f"time total averaged {time_total/run_count}")
