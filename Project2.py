import scipy.io
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



# read the data set
Numpyfile = scipy.io.loadmat('AllSamples.mat')
# print(Numpyfile["AllSamples"])
data = Numpyfile['AllSamples']
print(data[:5])
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data)
# print(scaled_data[:5])



# Strategy 1 initialization 1

variance1 = []

# set k range 2 to 10
for k in range(2,11):

    # instantiate KMeans
    kmeansRandom = KMeans(
        # initialization technique
        init="random",
        n_init=1,
        # k number
        n_clusters=k,
        random_state=42,
    )
    kmean = kmeansRandom.fit(data) # scaled_data
    variance1.append(kmean.inertia_)

print("variance", variance1)
plt.plot(range(2,11), variance1, 'bx-')
for (i, j) in zip(range(2,11), variance1):
    plt.text(i, j, f"({variance1[i-2]:.1f})")
plt.xlabel('Number of k')
plt.ylabel('variance')
plt.show()



# Strategy 1 initialization 2

variance2 = []

# set k range 2 to 10
for k in range(2,11):

    # instantiate KMeans
    kmeansRandom = KMeans(
        # initialization technique
        init="random",
        n_init=1,
        # k number
        n_clusters=k,
        random_state=45,
    )
    kmean = kmeansRandom.fit(data) # scaled_data
    variance2.append(kmean.inertia_)

print("variance", variance2)

# print plots
plt.plot(range(2,11), variance2, 'bx-')
for (i, j) in zip(range(2,11), variance2):
    plt.text(i, j, f"({variance2[i-2]:.1f})")
plt.xlabel('Number of k')
plt.ylabel('variance')
plt.show()



# Strategy 2 initialization 1

variance3 = []

# set k range 2 to 10
for k in range(2,11):

    # instantiate KMeans
    kmeansPP = KMeans(
        # initialization technique
        init="k-means++",
        n_init=1,
        # k number
        n_clusters=k,
        random_state=42,
    )
    kmean1 = kmeansPP.fit(data) # scaled_data
    variance3.append(kmean1.inertia_)

print("variance", variance3)

# print plots
plt.plot(range(2,11), variance3, 'bx-')
for (i, j) in zip(range(2,11), variance3):
    plt.text(i, j, f"({variance3[i-2]:.1f})")
plt.xlabel('Number of k')
plt.ylabel('variance')
plt.show()



# Strategy 2 initialization 2

variance4 = []

# set k range 2 to 10
for k in range(2,11):

    # instantiate KMeans
    kmeansPP = KMeans(
        # initialization technique
        init="k-means++",
        n_init=1,
        # k number
        n_clusters=k,
        random_state=45,
    )
    kmean1 = kmeansPP.fit(data) # scaled_data
    variance4.append(kmean1.inertia_)

print("variance", variance4)

# print plots
plt.plot(range(2,11), variance4, 'bx-')
for (i, j) in zip(range(2,11), variance4):
    plt.text(i, j, f"({variance4[i-2]:.1f})")
plt.xlabel('Number of k')
plt.ylabel('variance')
plt.show()