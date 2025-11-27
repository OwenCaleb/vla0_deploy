import matplotlib.pyplot as plt
import torch


def tile_images(images):
    """
    Tile images into a single image 把多张图横着拼成一张长图
    :param images: list[Tensor], 每个 Tensor: (H, W, 3)
                   或 4D Tensor: (bs, H, W, 3)
    :return: Tensor of shape (max_H, sum_W, 3)
    """
    # 如果是 batch tensor，拆成 list
    if isinstance(images, torch.Tensor) and images.dim() == 4:
        images = list(images)

    for img in images:
        assert len(img.shape) == 3, f"img.shape: {img.shape}"
        assert img.shape[2] == 3, f"img.shape: {img.shape}"

    # 收集所有图像的 (H_i, W_i)
    heights, widths = zip(*(im.shape[:-1] for im in images))
    total_width = sum(widths)  # 横向总宽度 = 各图宽度相加
    max_height = max(heights)  # 高度 = 所有图中最高的那一张

    # 先开一张足够大的黑底图，把每个子图依次贴上去。
    device = images[0].device
    dst = torch.zeros((max_height, total_width, 3), device=device)

    current_x = 0
    for i, img in enumerate(images):
        h, w, _ = img.shape
        dst[:h, current_x : current_x + w, :] = img
        current_x += w

    return dst


# ====== DEMO：造三块纯色小图，拼在一起 ======
# 红色块
img1 = torch.zeros(50, 80, 3)
img1[..., 0] = 1.0  # R

# 绿色块
img2 = torch.zeros(60, 100, 3)
img2[..., 1] = 1.0  # G

# 蓝色块
img3 = torch.zeros(40, 60, 3)
img3[..., 2] = 1.0  # B

images = [img1, img2, img3]

tiled = tile_images(images)

print("img1:", img1.shape)  # torch.Size([50, 80, 3])
print("img2:", img2.shape)  # torch.Size([60, 100, 3])
print("img3:", img3.shape)  # torch.Size([40, 60, 3])
print("tiled:", tiled.shape)  # torch.Size([60, 240, 3]) = (max_H, sum_W, 3)

# 可视化
plt.imshow(tiled.numpy())
plt.axis("off")
plt.show()
