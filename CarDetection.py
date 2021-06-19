import higra as hg
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage import filters
import time

def classificationEvaluation(label_gtr, label_classif):
    Nb = label_classif.shape[0]
    label_gtr = label_gtr.astype(int)
    label_classif = label_classif.astype(int)
    # print(label_gtr[0])
    nb_class = len(np.unique(label_gtr))

    ConfMat = np.zeros((nb_class, nb_class))

    for i in range(0, Nb):
        ConfMat[label_gtr[i], label_classif[i]] = ConfMat[label_gtr[i], label_classif[i]] + 1

    po = np.sum(np.diag(ConfMat)) / Nb
    pe = 0

    for i in range(0, nb_class):
        pe = pe + np.sum(ConfMat[:, i] * np.sum(ConfMat[i, :]))
    pe = pe / Nb ** 2

    oac = po
    kappa = (po - pe) / (1 - pe)

    perclass_CA = np.zeros((nb_class))
    for i in range(0, nb_class):
        perclass_CA[i] = ConfMat[i, i] / np.sum(ConfMat[i, :])

    aac = np.mean(perclass_CA)

    return oac, kappa, perclass_CA, aac, ConfMat
#------------------------------------------------------------------"
# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {1 : (255, 255, 255), # Impervious surfaces (white)
           2 : (0, 0, 255),     # Buildings (blue)
           3 : (0, 255, 255),   # Low vegetation (cyan)
           4 : (0, 255, 0),     # Trees (green)
           5 : (255, 255, 0),   # Cars (yellow)
           6 : (255, 0, 0),     # Clutter (red)
           7 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

####################################################################"
def carDetect(inputImg, treeType):
    start_time = time.time()
    if treeType == "max":
        preprocessImg = inputImg
    if treeType == "min":
        imgMax = np.max(inputImg)
        preprocessImg = imgMax - inputImg
    if treeType == "med":
        imgMed = np.median(inputImg)
        print(imgMed)
        preprocessImg = np.abs(inputImg - imgMed)

    print("Start building tree: " + treeType + " ...")
    if treeType=="tos":
        treeStruct, nodeAlt = hg.component_tree_tree_of_shapes_image2d(inputImg)
    else:
        img_graph = hg.get_4_adjacency_graph(inputImg.shape)
        treeStruct, nodeAlt = hg.component_tree_max_tree(img_graph, preprocessImg)
    print("End building tree: " + treeType)
    time_tree = time.time() - start_time
    numTreeNodes = treeStruct.num_vertices() - treeStruct.num_leaves()

    start_time = time.time()
    node_area = hg.attribute_area(treeStruct)
    node_compact = hg.attribute_compactness(treeStruct)
    # node_contour = hg.attribute_contour_length(treeStruct)
    # node_gaussian = hg.attribute_gaussian_region_weights_model(treeStruct, preprocessImg)
    # area2contour = node_area/node_contour
    time_attribute = time.time() - start_time

    start_time = time.time()
    area_upper = 2200
    area_lower = 1600
    selected_nodes = [n for n in range(treeStruct.num_vertices()) if node_area[n]>area_lower and node_area[n]<area_upper \
                 # and area2contour[n]>2 and area2contour[n]<5 \
                 and node_compact[n]>0.05 and node_compact[n]<0.15]

    # print(max_area.shape, nodeAlt.shape, len(selected_nodes), selected_nodes)
    # print('max_compact', np.min(max_compact[selected_nodes]), np.max(max_compact[selected_nodes]))
    # print('max_contour', np.min(max_contour[selected_nodes]), np.max(max_contour[selected_nodes]))
    # print('max_gaussian', np.min(max_gaussian[1]), np.max(max_gaussian[1]))
    # print('area2contour', np.min(area2contour), np.max(area2contour))

    car_nodes = np.zeros(nodeAlt.shape, dtype=bool)  # node label = 1 for cars
    condition = np.ones(nodeAlt.shape, dtype=bool)   # condition to propagate car label

    print("Start processing car nodes: " + treeType + " ...")
    for n in treeStruct.leaves_to_root_iterator(include_leaves = False, include_root = True):
        if n in selected_nodes:
            # print('n',n)
            car_nodes[n] = True
            for a in treeStruct.ancestors(n):
                # print(a)
                condition[a] = False

    all_car_nodes = hg.propagate_sequential(treeStruct, car_nodes, condition).copy()
    car_map = all_car_nodes[:treeStruct.num_leaves()].reshape(inputImg.shape[0],inputImg.shape[1]).copy()
    # print(car_map.shape, np.sum(car_map), np.unique(all_car_nodes), all_car_nodes.dtype, all_car_nodes.shape)
    # print(condition.shape, condition)
    img_filt_post = car_map  # dummy value
    print("End processing car nodes: " + treeType)
    time_detect = time.time() - start_time

    # print("Start car filter: " + treeType + " ...")
    # img_filt = hg.reconstruct_leaf_data(treeStruct, nodeAlt, condition)
    # if treeType == "min":
    #     img_filt_post = imgMax - img_filt
    # elif treeType == "med":
    #     img_gaussian = filters.gaussian(inputImg, sigma=np.sqrt(area_lower), preserve_range=True)  # smoothing
    #     med_sign_gaussian = np.sign(inputImg - imgMed)
    #     img_filt_post = imgMed + med_sign_gaussian * img_filt
    # else:
    #     img_filt_post = img_filt
    # print("End car filter: " + treeType)

    return car_map, img_filt_post, numTreeNodes, time_tree, time_attribute, time_detect
####################################################################"

imgGray = io.imread('Data/Potsdam_gray.bmp')            # small image
# imgRGB = io.imread('Data/top_potsdam_7_7_RGB.bmp')    # large image
# r, g, b = imgRGB[:, :, 0], imgRGB[:, :, 1], imgRGB[:, :, 2]
# imgGray = np.int16(0.2989 * r + 0.5870 * g + 0.1140 * b)
print('imgGray:', imgGray.shape, np.min(imgGray), np.max(imgGray))

img_gt = io.imread('Data/Potsdam_label.bmp')          # small label
# img_gt = io.imread('Data/top_potsdam_7_7_label.bmp')    # large label
print('Ground Truth:', img_gt.shape)
img_size = imgGray.shape[0]*imgGray.shape[1]

car_gt = np.uint8(np.all(img_gt == np.array((255, 255, 0)).reshape(1, 1, 3), axis=2))
print('cars',car_gt.shape)

#------------------------------------------------------------------"

car_map_min, img_filt_min, nNodes, tTree, tAttribute, tDetect = carDetect(imgGray, 'min')
print("Performance calculation: min")
oac, kappa, perclass_min, aac, ConfMat = classificationEvaluation(car_gt.ravel(), car_map_min.ravel())
precision = ConfMat[1,1]/(ConfMat[0,1]+ConfMat[1,1])
recall = ConfMat[1,1]/(ConfMat[1,0]+ConfMat[1,1])
tSum = tTree + tAttribute + tDetect
print("---> min(%d) precision(%.2f) recall(%.2f) time(%.2f , %.2f , %.2f , %.2f sec) "%(nNodes, 100*precision, 100*recall, tTree, tAttribute, tDetect, tSum))

car_map_max, img_filt_max, nNodes, tTree, tAttribute, tDetect = carDetect(imgGray, 'max')
print("Performance calculation: max")
oac, kappa, perclass_max, aac, ConfMat = classificationEvaluation(car_gt.ravel(), car_map_max.ravel())
precision = ConfMat[1,1]/(ConfMat[0,1]+ConfMat[1,1])
recall = ConfMat[1,1]/(ConfMat[1,0]+ConfMat[1,1])
tSum = tTree + tAttribute + tDetect
print("---> max(%d) precision(%.2f) recall(%.2f) time(%.2f , %.2f , %.2f , %.2f sec) "%(nNodes, 100*precision, 100*recall, tTree, tAttribute, tDetect, tSum))

car_map_tos, img_filt_tos, nNodes, tTree, tAttribute, tDetect = carDetect(imgGray, 'tos')
print("Performance calculation: tos")
oac, kappa, perclass_tos, aac, ConfMat = classificationEvaluation(car_gt.ravel(), car_map_tos.ravel())
precision = ConfMat[1,1]/(ConfMat[0,1]+ConfMat[1,1])
recall = ConfMat[1,1]/(ConfMat[1,0]+ConfMat[1,1])
tSum = tTree + tAttribute + tDetect
print("---> tos(%d) precision(%.2f) recall(%.2f) time(%.2f , %.2f , %.2f , %.2f sec) "%(nNodes, 100*precision, 100*recall, tTree, tAttribute, tDetect, tSum))

car_map_med, img_filt_med, nNodes, tTree, tAttribute, tDetect = carDetect(imgGray, 'med')
print("Performance calculation: med")
oac, kappa, perclass_med, aac, ConfMat = classificationEvaluation(car_gt.ravel(), car_map_med.ravel())
precision = ConfMat[1,1]/(ConfMat[0,1]+ConfMat[1,1])
recall = ConfMat[1,1]/(ConfMat[1,0]+ConfMat[1,1])
tSum = tTree + tAttribute + tDetect
print("---> med(%d) precision(%.2f) recall(%.2f) time(%.2f , %.2f , %.2f , %.2f sec) "%(nNodes, 100*precision, 100*recall, tTree, tAttribute, tDetect, tSum))
print(ConfMat)

plt.subplot(3,2,1)
plt.imshow(imgGray, cmap='gray', vmin=0, vmax=255)
plt.title("Input image")
plt.axis('off')

plt.subplot(3,2,2)
plt.imshow(car_gt)
plt.title("Ground Truth")
plt.axis('off')
io.imsave('car_gt.png', 255*np.uint8(car_gt))

plt.subplot(3,2,3)
plt.imshow(img_filt_min, cmap='gray', vmin=0, vmax=255)
plt.title("Min filtered")
plt.axis('off')

plt.subplot(3,2,4)
plt.imshow(car_map_min)
plt.title("Min class acc: %.3f - %.3f"%(perclass_min[1],perclass_min[0]))
plt.axis('off')
io.imsave('car_map_min.png', 255*np.uint8(car_map_min))

plt.subplot(3,2,5)
plt.imshow(img_filt_max, cmap='gray', vmin=0, vmax=255)
plt.title("Max filtered")
plt.axis('off')

plt.subplot(3,2,6)
plt.imshow(car_map_max)
plt.title("Max class acc: %.3f - %.3f"%(perclass_max[1],perclass_max[0]))
plt.axis('off')
io.imsave('car_map_max.png', 255*np.uint8(car_map_max))

plt.figure() #=====================

plt.subplot(3,2,1)
plt.imshow(imgGray, cmap='gray', vmin=0, vmax=255)
plt.title("Input image")
plt.axis('off')

plt.subplot(3,2,2)
plt.imshow(car_gt)
plt.title("Ground Truth")
plt.axis('off')

plt.subplot(3,2,3)
plt.imshow(img_filt_med, cmap='gray', vmin=0, vmax=255)
plt.title("Med filtered")
plt.axis('off')

plt.subplot(3,2,4)
plt.imshow(car_map_med)
plt.title("Med class acc: %.3f - %.3f"%(perclass_med[1],perclass_med[0]))
plt.axis('off')
io.imsave('car_map_med.png', 255*np.uint8(car_map_med))

plt.subplot(3,2,5)
plt.imshow(img_filt_tos, cmap='gray', vmin=0, vmax=255)
plt.title("ToS filtered")
plt.axis('off')

plt.subplot(3,2,6)
plt.imshow(car_map_tos)
plt.title("ToS class acc: %.3f - %.3f"%(perclass_tos[1],perclass_tos[0]))
plt.axis('off')
io.imsave('car_map_tos.png', 255*np.uint8(car_map_tos))

plt.show()


