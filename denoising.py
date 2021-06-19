#import sap
import higra as hg
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time

# --------- Read image ---------
img_gray = io.imread('Data\Lenna_gray.jpg')
rows, cols = img_gray.shape
print(rows,cols, np.min(img_gray), np.max(img_gray))

# --------- Add noise ---------
noise_level = 0.1
salt_pixels = np.int(noise_level * rows * cols)
coord_s1 = np.random.randint(0, rows, salt_pixels)
coord_s2 = np.random.randint(0, cols, salt_pixels)

pepper_pixels = np.int(noise_level * rows * cols)
coord_p1 = np.random.randint(0, rows, pepper_pixels)
coord_p2 = np.random.randint(0, cols, pepper_pixels)

img_noisy = img_gray.copy()
img_noisy[coord_s1,coord_s2] = 255
img_noisy[coord_p1,coord_p2] = 0

area_thresh = 20
img_noisy_blur = filters.gaussian(img_noisy, sigma=np.sqrt(area_thresh), preserve_range=True)  # smoothing
se = np.ones((3,3))
img_noisy_med = filters.median(img_noisy,se)
print(np.min(img_noisy_blur), np.max(img_noisy_blur))
# plt.subplot(1,2,1)
# plt.imshow(img_noisy, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_blur, cmap='gray')
# plt.show()

# --------- Build trees ---------
img_graph = hg.get_4_adjacency_graph(img_gray.shape)
max_tree, maxAlt = hg.component_tree_max_tree(img_graph, img_noisy)
min_tree, minAlt = hg.component_tree_min_tree(img_graph, img_noisy)
print('max_tree', maxAlt.shape, np.min(maxAlt), np.max(maxAlt), max_tree.num_vertices(), max_tree.num_leaves())
#img1 = minAlt[0+2000:262144+2000].reshape((512,512))
tos_tree, tosAlt = hg.component_tree_tree_of_shapes_image2d(img_noisy)

med = np.median(img_noisy)
med_sign = np.sign(img_noisy - med)  # sign matrix without filtering
med_sign_med = np.sign(img_noisy_med - med)
med_sign_blur = np.sign(img_noisy_blur - med)
med_sign_ideal = np.sign(img_gray - med)

img_med = np.uint8(np.abs(img_noisy - med))
med_tree, medAlt = hg.component_tree_max_tree(img_graph, img_med)
print('median', medAlt.shape, np.min(medAlt), np.max(medAlt), med_tree.num_vertices(), med_tree.num_leaves(), med_sign_blur.shape)

# --------- Filter tree ---------
area_max = hg.attribute_area(max_tree)
img_filt_max = hg.reconstruct_leaf_data(max_tree, maxAlt, area_max<area_thresh)
area_min = hg.attribute_area(min_tree)
img_filt_min = hg.reconstruct_leaf_data(min_tree, minAlt, area_min<area_thresh)
area_tos = hg.attribute_area(tos_tree)
img_filt_tos = hg.reconstruct_leaf_data(tos_tree, tosAlt, area_tos<area_thresh)
area_med = hg.attribute_area(med_tree)
img_filt_med = hg.reconstruct_leaf_data(med_tree, medAlt, area_med<area_thresh)

# ========== Max-tree + Min-tree ==========
min_max_tree, minMaxAlt = hg.component_tree_min_tree(img_graph, img_filt_max)
area_min_max = hg.attribute_area(min_max_tree)
img_filt_min_max = hg.reconstruct_leaf_data(min_max_tree, minMaxAlt, area_min_max<area_thresh)

# ========================================================================================
# ======== Test to see all the leaves children of a node have the same gray level ========
num_parents=0
for n in med_tree.leaves_to_root_iterator(include_leaves = False, include_root = True):
    leaf_levels = [medAlt[m] for m in med_tree.children(n) if med_tree.is_leaf(m)]
    # print('%d: #Childrens:%2d, #Leaves:%2d  Level %d = %d'%(n, med_tree.num_children(n), len(leaf_levels), np.unique(leaf_levels), medAlt[n]))
    if len(np.unique(leaf_levels))>1 or len(np.unique(leaf_levels))==0:
        print('================= Not unique =================')
    if len(np.unique(leaf_levels))==1:
        num_parents += 1
print('Num parents: %d - %d'%(num_parents, med_tree.num_vertices() - med_tree.num_leaves()))

# ========================================================================================
# Assign a sign to each tree node
# problem: children can have different sign (dissimilar to gray levels)
# select sign of each node as the majority sign of children

nNotUnique=0
notUniqueParents = []
medAltSgn = np.ones(medAlt.shape)      # sign of each tree node
medAltSgn[0:med_tree.num_leaves()] = med_sign.flatten()  # first leaves are pixels
for n in med_tree.leaves_to_root_iterator(include_leaves = False, include_root = True):
    leaf_signs = [medAltSgn[m] for m in med_tree.children(n) if med_tree.is_leaf(m)]
    leaf_unique = np.unique(leaf_signs)  # unique gray level but not unique sign
    #medAltSgn[n] = leaf_unique[0]
    medAltSgn[n] = np.sign(np.sum(leaf_signs))
    if len(leaf_unique) > 1:
        #print(leaf_unique)
        nNotUnique += 1
        notUniqueParents.append(n)
print('Not unique children: %d out of %d parents'%(nNotUnique,med_tree.num_vertices()-med_tree.num_leaves()))
print(notUniqueParents)
# print(med_tree.is_leaf(0),medAlt[1000],img_med[1,488])
# a = med_tree.is_leaf(range(262144-100, 262144+100))

# reconstruct sign matrix from tree structure (not good), This command removes all the leaves
med_sign_filt = hg.reconstruct_leaf_data(med_tree, medAltSgn, area_med<area_thresh)
# print(img_filt_med.shape, np.min(img_filt_med), np.max(img_filt_med))
# print(med_sign_filt.shape, np.unique(med_sign_filt))

# modify sign matrix sequentially from root to leaves
print(medAltSgn.shape, area_med.shape, med_tree.num_vertices())
deleted_nodes = area_med<area_thresh
for n in med_tree.root_to_leaves_iterator(include_leaves = True, include_root = True):
    pn = med_tree.parent(n)
    if not med_tree.is_leaf(n):
        if deleted_nodes[n]:
            medAltSgn[n] = medAltSgn[pn]
    else:
        if deleted_nodes[pn]:
            medAltSgn[n] = medAltSgn[pn]

med_sign_filt = medAltSgn[:med_tree.num_leaves()].reshape(512,512)
img_filt_med_reconst_original = med + med_sign * img_filt_med
img_filt_med_reconst_modify = med + med_sign_filt * img_filt_med

remain_noise = np.abs(img_filt_med_reconst_modify - img_gray)>100
#remain_noise = med_sign_filt == med_sign_ideal
print('remain_noise', np.min(remain_noise), remain_noise.shape)
plt.imshow(remain_noise, cmap='gray')
#plt.show()

nNoiseNotUnique = 0
for pixel, noise in enumerate(remain_noise.flatten()):
    if noise:
        # print(pixel, noise)
        pn = med_tree.parent(pixel)
        while deleted_nodes[pn]:
            pn = med_tree.parent(pn)
        if pn in notUniqueParents:
            nNoiseNotUnique += 1
            # print(pn)

print('remaining noise', np.sum(np.uint8(remain_noise)), nNoiseNotUnique)
# ========================================================================================

#img_diff = img_med - img_filt_med
#med_sign[img_diff>100] = - med_sign[img_diff>100]
#img_diff_flat = img_diff.flatten()
#change_indices = np.array(np.where(img_diff_flat != 0))
#change_indices = change_indices.ravel()

# med_sign_flat = med_sign_blur.flatten()  # first leaves are pixels
# print(img_diff_flat.shape, med_sign_flat.shape, area_med.shape)
#
# for chIdx in change_indices:
#     change_ancestors = med_tree.ancestors(chIdx)  # ancestors of the changed pixel
#     #print(chIdx, med_sign_flat[chIdx], change_ancestors)
#     remain_acestors_idx = np.array(np.where(area_med[change_ancestors]>=area_thresh)).ravel()
#     #print(area_med[change_ancestors])
#     first_remain_ancestor = change_ancestors[remain_acestors_idx[0]]
#     #print(remain_acestors_idx[0], first_remain_ancestor)
#     first_ancestor_childen = med_tree.children(first_remain_ancestor)
#     #print(first_ancestor_childen)
#     first_ancestor_leaf_childen_idx = med_tree.is_leaf(first_ancestor_childen)
#     #print(first_ancestor_leaf_childen_idx)
#     first_ancestor_leaf_childen = first_ancestor_childen[first_ancestor_leaf_childen_idx]
#     #print(first_ancestor_leaf_childen)
#     #print(np.unique(medAlt[first_ancestor_leaf_childen]))
#     #print(np.sum(med_sign_flat[first_ancestor_leaf_childen]))
#     med_sign_flat[chIdx] = np.sign(np.sum(med_sign_flat[first_ancestor_leaf_childen]))
#
# med_sign_modify = med_sign_flat.reshape(rows,cols)


# change_pixels = np.where(img_diff != 0)
# x_changed = change_pixels[0]
# y_changed = change_pixels[1]
# print((x_changed),(y_changed))
# x_img, y_img = img_diff.shape
# change_indices = [y_img*x_changed[i]+y_changed[i] for i in range(len(x_changed))]
# print(len(change_indices))

img_filt_med_reconst_blur = med + med_sign_blur * img_filt_med
# img_filt_med_modify_blur = med_blur + med_sign_modify * img_filt_med
img_filt_med_reconst_med = med + med_sign_med * img_filt_med
# img_filt_med_modify_med = med_med + med_sign_modify * img_filt_med


#============  save images ==============
# io.imsave('lenna_original.png', img_gray)
# io.imsave('lenna_noisy.png', img_noisy)
# io.imsave('lenna_max.png', img_filt_max)
# io.imsave('lenna_min.png', img_filt_min)
# io.imsave('lenna_tos.png', img_filt_tos)
# io.imsave('lenna_median.png', img_filt_med_reconst_blur)

#============  Metric ==============
img_filt_med_reconst_blur = np.uint8(img_filt_med_reconst_blur)
#print('Origial PSNR %0.2f'%(peak_signal_noise_ratio(img_gray,img_gray)))
noisy_PSNR = peak_signal_noise_ratio(img_gray,img_noisy)
print('Noisy PSNR: %0.2f'%(noisy_PSNR))
print('Max-tree PSNR: %0.2f'%(peak_signal_noise_ratio(img_gray,img_filt_max)))
print('Min-tree PSNR: %0.2f'%(peak_signal_noise_ratio(img_gray,img_filt_min)))
print('ToS PSNR: %0.2f'%(peak_signal_noise_ratio(img_gray,img_filt_tos)))
median_blur_PSNR = peak_signal_noise_ratio(img_gray,np.uint8(img_filt_med_reconst_blur))
print('Median_blur PSNR: %0.2f'%(median_blur_PSNR))
median_original_PSNR = peak_signal_noise_ratio(img_gray,np.uint8(img_filt_med_reconst_original))
print('Median_original PSNR: %0.2f'%(median_original_PSNR))
median_modify_PSNR = peak_signal_noise_ratio(img_gray,np.uint8(img_filt_med_reconst_modify))
print('Median_modify PSNR: %0.2f'%(median_modify_PSNR))
print()
print('Origial ssim: %0.2f'%(structural_similarity(img_gray,img_gray)))
noisy_ssim = structural_similarity(img_gray,img_noisy)
print('Noisy ssim: %0.2f'%(noisy_ssim))
print('Max-tree ssim: %0.2f'%(structural_similarity(img_gray,img_filt_max)))
print('Min-tree ssim: %0.2f'%(structural_similarity(img_gray,img_filt_min)))
print('ToS ssim: %0.2f'%(structural_similarity(img_gray,img_filt_tos)))
median_blur_ssim = structural_similarity(img_gray,np.uint8(img_filt_med_reconst_blur))
print('Median_blur ssim: %0.2f'%(median_blur_ssim))
median_original_ssim = structural_similarity(img_gray,np.uint8(img_filt_med_reconst_original))
print('Median_original ssim: %0.2f'%(median_original_ssim))
median_modify_ssim = structural_similarity(img_gray,np.uint8(img_filt_med_reconst_modify))
print('Median_modify ssim: %0.2f'%(median_modify_ssim))

#============  Plot ==============
plt.imshow(img_filt_med_reconst_modify, cmap='gray', vmin=0, vmax=255)
#plt.show()

plt.subplot(2,2,1)
plt.imshow(img_noisy, cmap='gray', vmin=0, vmax=255)
plt.title('Noisy Image\n(PSNR:%.2f , ssim:%.2f)'%(noisy_PSNR, noisy_ssim))
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(img_filt_med_reconst_original, cmap='gray', vmin=0, vmax=255)
plt.title('Filtered Image (original sign)\n(PSNR:%.2f , ssim:%.2f)'%(median_original_PSNR, median_original_ssim))
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(img_filt_med_reconst_modify, cmap='gray', vmin=0, vmax=255)
plt.title('Filtered Image (modified sign)\n(PSNR:%.2f , ssim:%.2f)'%(median_modify_PSNR, median_modify_ssim))
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(img_filt_med_reconst_blur, cmap='gray', vmin=0, vmax=255)
plt.title('Filtered Image (blurred sign)\n(PSNR:%.2f , ssim:%.2f)'%(median_blur_PSNR, median_blur_ssim))
plt.axis('off')
plt.show()


# plt.subplot(2,2,1)
# plt.imshow(img_noisy, cmap='gray', vmin=0, vmax=255)
# plt.subplot(2,2,2)
# plt.imshow(img_med, cmap='gray', vmin=0, vmax=255)
# #io.imsave('lenna_median_noisy.png', img_med)
# plt.subplot(2,2,3)
# plt.imshow(img_filt_med, cmap='gray', vmin=0, vmax=255)
# #io.imsave('lenna_median_filter.png', img_filt_med)
# plt.subplot(2,2,4)
# plt.imshow(img_filt_med_reconst2, cmap='gray', vmin=0, vmax=255)
# #io.imsave('img_filt_med_reconst2.png', img_filt_med_reconst2)
# #io.imsave('lenna_median_reconstruct.png', img_filt_med_reconst_blur)
# plt.show()

# plt.subplot(1,3,1)
# plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
# plt.subplot(1,3,2)
# plt.imshow(img_filt_min_max, cmap='gray', vmin=0, vmax=255)
# plt.subplot(1,3,3)
# plt.imshow(img_filt_tos, cmap='gray', vmin=0, vmax=255)
# plt.show()

# plt.subplot(2,4,1)
# plt.imshow(img_noisy, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('(1) Noisy image')
# plt.subplot(2,4,5)
# plt.imshow(img_med, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('(2) Median image')
#
# plt.subplot(2,4,2)
# plt.imshow(img_filt_med, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('(3) Filtered median')
# plt.subplot(2,4,6)
# plt.imshow(img_filt_tos, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('(4) Filtered ToS')
#
# plt.subplot(2,4,3)
# plt.imshow(img_filter_blur, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('(5) Blured image')
# plt.subplot(2,4,7)
# plt.imshow(img_filt_med_reconst_blur, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('(6) Median Reconstructed (blur)')
#
# plt.subplot(2,4,4)
# plt.imshow(img_filter_med, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('(7) Median filtering (5,5)')
# plt.subplot(2,4,8)
# plt.imshow(img_filt_med_reconst_med, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('(8) Median Reconstructed (med)')
#
# plt.show()

# plt.subplot(3,2,1)
# plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('Original image')
# plt.subplot(3,2,2)
# plt.imshow(img_noisy, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('Noisy image')
#
# plt.subplot(3,2,3)
# plt.imshow(img_filt_max, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('Max-tree filtering')
# plt.subplot(3,2,4)
# plt.imshow(img_filt_min, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('Min-tree filtering')
#
# plt.subplot(3,2,5)
# plt.imshow(img_filt_tos, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('Tree-of-Shape filtering')
# plt.subplot(3,2,6)
# plt.imshow(img_filt_med_reconst_blur, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.title('Median-tree filtering')
#
# plt.show()