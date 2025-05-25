import torch 
import random
import os

class visualise_dataset():
    def plot_raw_images():
        images_path_list = list(config["data_path"].glob("*/*.npy"))
        random_image_path = random.sample(images_path_list,1)
        for img_path in random_image_path:
            img = np.load(img_path, allow_pickle=True)
            img = torch.from_numpy(img).unsqueeze(0)
            print(f"image shape {img.shape}")
            print(f"Image height: {img.shape[1]}") 
            print(f"Image width: {img.shape[2]}")
            print(f"Image min: {img.min():.3f}")
            print(f"Image max: {img.max():.3f}")

            plt.imshow(img.permute(1,2,0)) 
            plt.axis("off") 
            plt.show()
    
    def plot_transformed_images(dataset,n,seed=None):
        if n>10:
            n=10
        if seed:
            torch.manual_seed(seed)
        random_samples_idx = random.sample(range(len(dataset.file_paths)),n)
        cols = min(n,5)
        rows = math.ceil(n/cols)
        fig, axes = plt.subplots(rows,cols,figsize = (cols*3,rows*3))
        axes = axes.flatten() if n > 1 else [axes]

        for i, idx in enumerate(random_samples_idx):
            image = dataset[idx]
            axes[i].imshow(image.permute(1, 2, 0))
            axes[i].axis("off")
            axes[i].set_title(f"Shape: {image.shape}", fontsize=8)
        
        plt.tight_layout()

