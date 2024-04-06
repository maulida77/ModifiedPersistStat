import numpy as np
from PIL import Image
import gudhi as gd
import matplotlib.pyplot as plt
# from bokeh.models import ColumnDataSource, Range1d
# from bokeh.plotting import figure
# import matplotlib.pyplot as plt
# import scipy
# from scipy import ndimage
# import PIL
# from persim import plot_diagrams
# from ripser import ripser, lower_star_img
from persim import plot_diagrams



i=1

array= np.array([[[ 4, 4, 4],[ 2, 2, 2],[ 4, 4, 4]],
                [[ 1, 1, 1],[ 5, 5, 5],[ 1, 1, 1]],
                [[ 1, 1, 1], [ 2, 2, 2],[ 4, 4, 4]]])

array = np.array(array, dtype=np.uint8)
img = Image.fromarray(array)
# plt.imshow(img)
# plt.show()
img.save('img.png')
data = Image.open('img.png')
data = data.convert("L")
# plt.imshow(data)
# plt.show()
data = np.array(data)
print(data)
dim1=3
dim2=3
# data_gudhi = np.resize(data, [dim1, dim2])
data_gudhi = data.reshape(dim1*dim2,1)
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

np.savetxt("l_d0_"+str(i), pd0_l)
np.savetxt("l_d1_"+str(i), pd1_l)
np.savetxt("u_d0_"+str(i), pd0_u)
np.savetxt("u_d1_"+str(i), pd1_u)


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(data)
clb=plt.colorbar(location='left', ticks=[1, 2, 3, 4, 5])
clb.ax.tick_params(labelsize=18) 
plt.xticks([])
plt.yticks([])
# plt.title("Image", fontsize = 24)

plt.subplot(122)
plot_diagrams(pd0_l)
plt.title("0-D Persistence Diagram")
plt.tight_layout()
plt.show()
plt.savefig('save.png')




# ************************************************
#         Visualizing Persistence Barcodes
# ************************************************

# # Visualizing PersBarcode in dim-0 (Together with its error handling)
# if len(pd0_l) != 0:
#     source = ColumnDataSource(data={'birth': pd0_l[:,0], 'death': pd0_l[:,1], 'y': range(len(pd0[:,0]))})
# else:
#     # This is an error handling step so that when the PersBarcode data is empty, 
#     # then initialize an empty array and return an empty plot to avoid getting an error
#     # A Similar approach has been repeated for the rest of the vectorization method visualization.
#     source = ColumnDataSource(data={'birth': [], 'death': [], 'y': []})
            
# fig = figure(title='Persistence Barcode [dim = 0]', height=250, tools = tools)
# fig.hbar(y='y', left ='birth', right='death', height=0.1, alpha=0.7, source=source)
# fig.yaxis.visible = False

# if len(pd0_l) == 1:
#     fig.y_range = Range1d(-1, 1)
#     st.bokeh_chart(fig, use_container_width=True)

#             # Visualizing PersBarcode in dim-1
#             if len(pd1) != 0:
#                 source = ColumnDataSource(data={'birth': pd1[:,0], 'death': pd1[:,1], 'y': range(len(pd1[:,0]))})
#             else:
#                 source = ColumnDataSource(data={'birth': [], 'death': [], 'y': []})
#             fig = figure(title='Persistence Barcode [dim = 1]', height=250, tools = tools)
#             fig.hbar(y='y', left ='birth', right='death', height=0.1, color="darkorange", alpha=0.7, source=source)
#             fig.yaxis.visible = False

#             if len(pd1) == 1:
#                 fig.y_range = Range1d(-1, 1)
            
#             st.bokeh_chart(fig, use_container_width=True)

#             create_download_button('PH barcode dim0', pd0)
#             create_download_button('PH barcode dim1', pd1)
#             v_space(2)

#         # ************************************************
#         #         Visualizing Persistence Diagram
#         # ************************************************
#         if visualizationMode == 'Persistence Diagram':
#             st.subheader("Persistence Diagram")
#             # dgms = []

#             # if len(pd0) != 0:
#             #     dgms.append(pd0)
            
#             # if len(pd1) != 0:
#             #     dgms.append(pd1)

#             col1, col2 = st.columns(2)

#             fig, ax = plt.subplots()

#             if len(pd0) != 0:
#                 persim.plot_diagrams([pd0],labels= ["H0"], ax=ax)
            
#             ax.set_title("Persistence Diagram H0")
#             col1.pyplot(fig)

#             fig, ax = plt.subplots()
            
#             if len(pd1) != 0:
#                 persim.plot_diagrams([pd1],labels= ["H1"], ax=ax)
            
#             ax.set_title("Persistence Diagram H1")
#             col2.pyplot(fig)
        # Create a download button for PersDiagram plot, PLease.















# def random_img(output, width, height):

#     array = np.random.random_integers(0,255, (height,width,3))  
#     array = np.array(array, dtype=np.uint8)
#     img = Image.fromarray(array)
#     img.save(output)

# random_img('random.png', 3, 8)




# img1 = np.matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]])
# img2 = np.matrix([[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]])
# plt.imshow(img1)
# # plt.show()
# plt.savefig('img1.png')
# # array = np.random.random_integers(0,255, (8,3,3))
# # print(array)