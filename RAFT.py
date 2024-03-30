import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.io import read_video
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
import torchvision.transforms as transforms

plt.rcParams["savefig.bbox"] = "tight"
weights = Raft_Large_Weights.DEFAULT
transforms_of = weights.transforms()

device = "cuda" if torch.cuda.is_available() else "cpu"


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show(block=False)

def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    return transforms_of(img1_batch, img2_batch)

def main():
    MINE = True
    if MINE:
        image_1_orig = cv2.cvtColor(cv2.imread('my_images/1.jpg'), cv2.COLOR_BGR2RGB)
        image_2_orig = cv2.cvtColor(cv2.imread('my_images/2.jpg'), cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_1_tensor = transform(image_1_orig)
        image_2_tensor = transform(image_2_orig)

        img1_batch = torch.stack([image_1_tensor])
        img2_batch = torch.stack([image_2_tensor])


    else:
        video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
        video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
        _ = urlretrieve(video_url, video_path)

        frames, _, _ = read_video(str(video_path), output_format="TCHW")

        # img1_batch = torch.stack([frames[100], frames[150]])
        # img2_batch = torch.stack([frames[101], frames[151]])

        img1_batch = torch.stack([frames[100]])
        img2_batch = torch.stack([frames[110]])

    plot(img1_batch)

    img1_batch_preproc, img2_batch_preproc = preprocess(img1_batch, img2_batch)
    print(f"shape = {img1_batch_preproc.shape}, dtype = {img2_batch_preproc.dtype}")

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    list_of_flows = model(img1_batch_preproc.to(device), img2_batch_preproc.to(device))
    print(f"type = {type(list_of_flows)}")
    print(f"length = {len(list_of_flows)} = number of iterations of the model")

    predicted_flows = list_of_flows[-1]
    print(f"dtype = {predicted_flows.dtype}")
    print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
    print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

    flow_imgs = flow_to_image(predicted_flows)
    flow_np = predicted_flows.detach().numpy()

    # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
    img1_batch_preproc = [(img1 + 1) / 2 for img1 in img1_batch_preproc]

    grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch_preproc, flow_imgs)]
    plot(grid)


    '''plot numpy'''
    def normalize_image(image):
        normalized_image = image - np.min(image)
        normalized_image = ((normalized_image/np.max(normalized_image))*255).astype(np.uint8)
        return normalized_image

    u_image_normalized = normalize_image(flow_np[0,0,:,:])
    v_image_normalized = normalize_image(flow_np[0,1,:,:])


    fig, axs = plt.subplots(1,2)
    axs[0].imshow(u_image_normalized)
    axs[1].imshow(v_image_normalized)
    plt.show(block=False)


    return


if __name__ == "__main__":
    main()
