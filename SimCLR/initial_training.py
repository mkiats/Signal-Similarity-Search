import os
import random
import numpy as np
import matplotlib.pyplot as plt
import gc

from scipy.io import wavfile

import torch

from model import tensorify, augment, ResNet, loss_function

# augments n images twice each and returns a dictionary of the produced tensors, where dict["spec1"][i] and dict["spec2"][i] are the  positive pairs 
def batch(root_dir, n=32):
    files = os.listdir(root_dir)
    files = [file for file in files if file.endswith('.wav')]
    
    l = len(files)
    random.shuffle(files)
    
    for i in range(0, l, n):
        dict = {}
        dict["spec1"] = []
        dict["spec2"] = []
        for j in range(n):
            index = i + j
            print(f"{index}/{l}", end="\r")
            if index >= l:
                return
            file_name = os.path.join(root_dir, files[index])
            
            sample_rate, data = wavfile.read(file_name)

            f, t, spec1 = augment(data, sample_rate)
            spec1 = tensorify(spec1)
            f, t, spec2 = augment(data, sample_rate)
            spec2 = tensorify(spec2)

            dict["spec1"].append(spec1)
            dict["spec2"].append(spec2)
        yield dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_size = 128
NUM_EPOCHS = 50

# change input dir accordingly to be passed into the batch function
input_dir = os.path.join("data", "unlabelled")
# train for the specified learning rates and batch sizes
for LR in [0.00005]:
    for BATCH_SIZE in [32]:
        
        # initialise model and optimiser
        model = ResNet(embedding_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        save_path = "lr" + str(LR).replace(".", "") + "b" + str(BATCH_SIZE)
        save_dir = "results"

        # set seeds
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        # get resnet in train mode
        model.train()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, "optimizers")):
            os.makedirs(os.path.join(save_dir, "optimizers"))

        losses = []
        # unsupervised training
        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0

            for bat in batch(input_dir, BATCH_SIZE):
                optimizer.zero_grad()

                combined_embedding1 = []
                for ts in bat['spec1']:
                    combined_embedding1.append(model(ts.to(device)))
                combined_embedding1 = torch.cat(combined_embedding1)

                combined_embedding2 = []
                for ts in bat['spec2']:
                    combined_embedding2.append(model(ts.to(device)))
                combined_embedding2 = torch.cat(combined_embedding2)

                loss = loss_function(combined_embedding1, combined_embedding2, 0.05)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                del combined_embedding1, combined_embedding2, loss
                torch.cuda.empty_cache()
                gc.collect()

            epoch_loss = running_loss / len(os.listdir(input_dir))
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f}")
            losses.append(epoch_loss)
            
            # save training weights
            torch.save(model.state_dict(), os.path.join(save_dir, save_path) + f"_{epoch + 1}.pth")
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizers", save_path) + f"_{epoch + 1}.pth")

        # Generate x-axis values (epochs or steps)
        epochs = range(1, len(losses) + 1)

        # Plot the loss graph
        plt.plot(epochs, losses, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f"Learning rate: {LR}, Batch size: {BATCH_SIZE}")

        fig_dir = os.path.join(save_dir, "figures")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        # save the graph
        plt.savefig(fig_dir + "/" + save_path + ".png")
        plt.clf()