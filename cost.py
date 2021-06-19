import sap
import higra as hg
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import rescale, resize
import time
from sys import getsizeof
from skimage import io, filters
#from memory_profiler import profile

# @profile
def show_sizeof(x, level=0):
    print("\t" * level, x.__class__, getsizeof(x), x)
    if hasattr(x, '__iter__'):
        if hasattr(x, 'items'):
            for xx in x.items():
                show_sizeof(xx, level + 1)
        else:
            for xx in x:
                show_sizeof(xx, level + 1)

# @profile
def myProfileCode():
    img_gray = io.imread('Data/Lenna_gray.jpg')
    # img_RGB = io.imread('Data/Penguins.jpg')
    # r, g, b = img_RGB[:, :, 0], img_RGB[:, :, 1], img_RGB[:, :, 2]
    # img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    # img_gray = resize(img_gray, (100,100))
    #print(img_gray.shape)
    # print(np.min(img_gray), np.max(img_gray))
    img_white = np.ones(img_gray.shape)*255
    #img_gray = img_white

    img_neg = 255 - img_gray

    # print(img_med.shape, np.unique(med_sign))
    # print(np.min(img_med), np.max(img_med))

    # plt.subplot(1,2,1)
    # plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(1,2,2)
    # plt.imshow(img_white, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    testObject = np.zeros(1,dtype=np.bool)
    testObject[0] = 1
    print('testObject', type(testObject), getsizeof(testObject))
    area_thresh = 2500

    #-------------- Max Tree --------------
    t0 = time.time()
    max_graph = hg.get_4_adjacency_graph(img_gray.shape)
    max_tree, maxAlt = hg.component_tree_max_tree(max_graph, img_gray)
    #max_tree, maxAlt = hg.binary_partition_tree_average_linkage(max_graph,edge_similarity_max)
    area_max = hg.attribute_area(max_tree)
    img_filt_max = hg.reconstruct_leaf_data(max_tree, maxAlt, area_max<area_thresh)
    print("\nMax Tree ----- %0.2f seconds -----" % (time.time() - t0))
    # show_sizeof([True, False, True, False])
    print("memory footpint -> ",type(max_tree.root()),getsizeof(max_tree.root()),type(maxAlt[1000]),getsizeof(maxAlt[1000]), img_filt_max.shape)
    print('vertices: %d , leaves: %d'%(max_tree.num_vertices(), max_tree.num_leaves()))

    edge_dissimilarity_max = hg.weight_graph(max_graph, img_gray, hg.WeightFunction.L1) + 1e-10
    print(np.min(edge_dissimilarity_max),np.max(edge_dissimilarity_max))
    edge_similarity_max = 1/(edge_dissimilarity_max)
    print(np.min(edge_similarity_max),np.max(edge_similarity_max))
    cost_dasgupta_max = hg.dasgupta_cost(max_tree, edge_dissimilarity_max, max_graph)
    cost_divergence_max = hg.tree_sampling_divergence(max_tree, edge_similarity_max, max_graph)
    print('cost_dasgupta_max: %0.4e , cost_divergence_max: %0.4e'%(cost_dasgupta_max, cost_divergence_max))

    #-------------- Median Tree --------------
    t1 = time.time()
    d1, d2 = img_gray.shape
    med = np.median(img_gray)
    img_med = np.uint8(np.abs(img_gray - med))
    img_filter_blur = filters.gaussian(img_gray, sigma=np.sqrt(area_thresh), preserve_range=True)  # smoothing
    med_sign = np.sign(img_filter_blur - med)

    med_graph = hg.get_4_adjacency_graph(img_med.shape)
    med_tree, medAlt = hg.component_tree_max_tree(med_graph, img_med)
    area_med = hg.attribute_area(med_tree)
    img_filt_med = hg.reconstruct_leaf_data(med_tree, medAlt, area_med<area_thresh)

    # === begin update sign
    # medAltSgn = np.zeros(medAlt.shape)  # sign of each tree node
    # medAltSgn[0:med_tree.num_leaves()] = med_sign.flatten()  # first leaves are pixels
    # for n in med_tree.leaves_to_root_iterator(include_leaves=False, include_root=True):
    #     leaf_signs = [medAltSgn[m] for m in med_tree.children(n) if med_tree.is_leaf(m)]
    #     medAltSgn[n] = np.sign(np.sum(leaf_signs))
    #
    # print('here')
    # deleted_nodes = area_med < area_thresh
    # medAltSgnModify = medAltSgn.copy()
    # for n in med_tree.root_to_leaves_iterator(include_leaves=True, include_root=True):
    #     pn = med_tree.parent(n)
    #     if not med_tree.is_leaf(n):
    #         if deleted_nodes[n]:
    #             medAltSgnModify[n] = medAltSgnModify[pn]
    #     else:
    #         if deleted_nodes[pn]:
    #             medAltSgnModify[n] = medAltSgnModify[pn]
    # med_sign = medAltSgnModify[:med_tree.num_leaves()].reshape(d1, d2)
    # === end update sign

    img_filt_med_reconst_blur = med + med_sign * img_filt_med

    print("\nMedian Tree ----- %0.2f seconds -----" % (time.time() - t1))
    print("memory footpint -> ",type(med_tree.root()),getsizeof(med_tree.root()),type(medAlt[1000]),getsizeof(medAlt[1000]))
    print('vertices: %d , leaves: %d'%(med_tree.num_vertices(), med_tree.num_leaves()))

    edge_dissimilarity_med = hg.weight_graph(med_graph, img_med, hg.WeightFunction.L1) + 1e-10
    edge_similarity_med = 1/(edge_dissimilarity_med)
    cost_dasgupta_med = hg.dasgupta_cost(med_tree, edge_dissimilarity_med, med_graph)
    cost_divergence_med = hg.tree_sampling_divergence(med_tree, edge_similarity_med, med_graph)
    print('cost_dasgupta_med: %0.4e , cost_divergence_med: %0.4e'%(cost_dasgupta_med, cost_divergence_med))

    #-------------- Tree of Shape --------------
    t2 = time.time()
    tos_tree, tosAlt = hg.component_tree_tree_of_shapes_image2d(img_gray)
    area_tos = hg.attribute_area(tos_tree)
    img_filt_tos = hg.reconstruct_leaf_data(tos_tree, tosAlt, area_tos<area_thresh)
    print("\nToS ----- %0.2f seconds -----" % (time.time() - t2))
    print("memory footpint -> ",type(tos_tree.root()),getsizeof(tos_tree),type(tosAlt[1000]),getsizeof(tosAlt))
    print('vertices: %d , leaves: %d'%(tos_tree.num_vertices(), tos_tree.num_leaves()))

    edge_dissimilarity_tos = hg.weight_graph(max_graph, img_gray, hg.WeightFunction.L1) + 1e-10
    edge_similarity_tos = 1/(edge_dissimilarity_tos)
    print(edge_similarity_tos.shape)
    cost_dasgupta_tos = hg.dasgupta_cost(tos_tree, edge_dissimilarity_tos, max_graph)
    cost_divergence_tos = hg.tree_sampling_divergence(tos_tree, edge_similarity_tos, max_graph)
    print('cost_dasgupta_tos: %0.4e , cost_divergence_tos: %0.4e'%(cost_dasgupta_tos, cost_divergence_tos))


    # plt.subplot(1,2,1)
    # plt.imshow(med_reconstruct, cmap='gray', vmin=0, vmax=255)
    # plt.title('Median [%d,%d]'%(np.min(med_reconstruct),np.max(med_reconstruct)))
    # plt.subplot(1,2,2)
    # plt.imshow(filtered_max, cmap='gray', vmin=0, vmax=255)
    # plt.title('Tree of shape [%d,%d]'%(np.min(filtered_max),np.max(filtered_max)))
    # plt.show()


myProfileCode()