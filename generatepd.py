import numpy as np
from PIL import Image
import gudhi as gd


path_diag = "Batik300/pdiagrams/" #"BatikNitik960/pdiagrams/" #"Outex-TC-00024/pdiagrams/"
index=range(300) #960 for batiknitik #2720 for Outex

for i in index:
    file_path = "Batik300/data/images/"+str(i)+".jpg"  # "BatikNitik960/data/images/"+str(i)+".jpg" # "Outex-TC-00024/data/images/"+str(i)+".jpg"

    data = Image.open(file_path)
    data = data.convert("L")
    data = np.array(data)
    dim1=128 # 512 for batiknitik # 128 for batik300 and outex 
    dim2=128 # 512 for batiknitik # 128 for batik300 and outex 
    data_gudhi = np.resize(data, [dim1, dim2])
    data_gudhi = data_gudhi.reshape(dim1*dim2,1)
    cub_filtration_l = gd.CubicalComplex(dimensions = [dim1,dim2], top_dimensional_cells=data_gudhi)
    cub_filtration_l.persistence()
    cub_filtration_u = gd.CubicalComplex(dimensions = [dim1,dim2], top_dimensional_cells=-data_gudhi)
    cub_filtration_u.persistence()

    pd0_l = cub_filtration_l.persistence_intervals_in_dimension(0)
    pd1_l = cub_filtration_l.persistence_intervals_in_dimension(1)
    pd0_u= cub_filtration_u.persistence_intervals_in_dimension(0)-1
    pd1_u= cub_filtration_u.persistence_intervals_in_dimension(1)-1

    for j in range(pd0_l.shape[0]):
        if pd0_l[j,1]==np.inf:
            pd0_l[j,1]=256
    for j in range(pd1_l.shape[0]):
        if pd1_l[j,1]==np.inf:
            pd1_l[j,1]=256

    for j in range(pd0_u.shape[0]):
        if pd0_u[j,1]==np.inf:
            pd0_u[j,1]=256
    for j in range(pd1_u.shape[0]):
        if pd1_u[j,1]==np.inf:
            pd1_u[j,1]=256

    np.savetxt(path_diag+"l_d0_"+str(i), pd0_l)
    np.savetxt(path_diag+"l_d1_"+str(i), pd1_l)
    np.savetxt(path_diag+"u_d0_"+str(i), pd0_u)
    np.savetxt(path_diag+"u_d1_"+str(i), pd1_u)

