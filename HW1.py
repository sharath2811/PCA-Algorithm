import numpy as np
from numpy import linalg as la
from matplotlib import*
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 


#Q1-------------------------
data = np.genfromtxt("magic04.data.txt", delimiter=',')
#np.array(data)
Mean = np.mean(data, axis=0) #center data
Center=data-Mean
my_cov=(np.dot(Center.T,Center))/Center.shape[0] #Calculate covariance
#df1 = data.replace(np.nan, 0, regex=True) #standardize data
#X_std = StandardScaler().fit_transform(df1)
cov_mat=np.cov(Center, rowvar=False)
#PleaseWork = np.dot(Center.T, Center)
np.array_equal(my_cov,cov_mat) #checks if both matrices are equal


#Q2

cov_mat[np.isnan(cov_mat)] = 0
eiVal,eiVec= la.eig(cov_mat)
ei1=eiVec[0] #largest eigen vectors
ei2=eiVec[1]
PVec= (np.dot(ei1,ei1.T))+(np.dot(ei2,ei2.T))   #Projection vector
result=Pvec*Center #projection of centered vector into the subspace spanned by the two eigenvectors
result_cov=np.cov(result)#covariance of result

#eig_pairs = [(np.abs(eiVal[i]), eiVec[:,i]) for i in range(len(eiVal))] #sorted eigen vectors and eigen values
#eig_pairs.sort(key=lambda x: x[0], reverse=True)

#W= np.hstack((eig_pairs[0][1].reshape(11,11), eig_pairs[1][1].reshape(11,11)))

#PCA

#Q3

I=np.identity(11)
A=I*eiVal
decomp_var=eiVec*A*eiVal #decomposed variance

#Q4

for in range(0,2):
    for i in eig_pairs:
        total_var=total_var+i[0]
    print("okay");
    
for i in eig_pairs:
        total_var=total_var+i[0]

        
def PCA(){
total_var=0
for i in eig_pairs: total_var=total_var+i[0]
calc_var=6579.79139978+3853.87048182;

fVar=total_var/calc_var #fraction of total variance

U=[ei1,ei2]
A=U.T*data
    }
pca = PCA(n_components=2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
