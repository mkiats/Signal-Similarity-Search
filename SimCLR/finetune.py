import pandas as pd
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import gc

from scipy.io import wavfile

import torch

from model import tensorify, augment, ResNet, loss_function

''' Given a data_dir and the pickle_dir, iterates through all train_dirs under the data_dir and batches the signals according to their labels. Each signal is paired with another signal of the same type to form a positive pair, and the batch size is simply the number of signal types.'''
def labelled_batch(train_dirs, data_dir, pickle_dir):
    label_groups = {}
    
    for folder in train_dirs:
        pickle_path = pickle_dir + folder + ".pickle"
        file_dir = data_dir + folder
        
        obj = pd.read_pickle(pickle_path)
    
        files = os.listdir(file_dir)
        
        for file in files:
            file_path = os.path.join(file_dir, file)
            labels = obj['GTLabelnamesAll'][os.path.splitext(file)[0]]

            for label in labels:
                if label in label_groups:
                    label_groups[label].append(file_path)
                else:
                    label_groups[label] = [file_path]
    
    # remove signal types with less than 10 samples to prevent overfitting
    label_groups = {key: value for key, value in label_groups.items() if len(value) > 10}
    
    cur_idx = {key: 0 for key in label_groups.keys()}
    sizes = {key: len(value) for key, value in label_groups.items()}
    total_size = sum(sizes.values())
    
    max_size = max(value for value in sizes.values())
    
    # iterate until the most frequently occuring signal type has been fully processed
    for i in range(max_size//2):
        dict = {}
        dict["spec1"] = []
        dict["spec2"] = []
        for signal_type in label_groups:
            # for each signal type, get 2 samples to form a positive pair
            for j in range(2):   
                idx = i*2+j
                # if a signal type has been fully processed, shuffle the files and restart
                if idx % sizes[signal_type] == 0:
                    random.shuffle(label_groups[signal_type])

                file = label_groups[signal_type][idx % sizes[signal_type]]

                sample_rate, data = wavfile.read(file)
                f, t, spec = augment(data, sample_rate)
                spec = tensorify(spec)
                dict["spec" + str(j + 1)].append(spec)
        print(f"{i+1}/{max_size//2}", end="\r")
        yield dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pretrained model
LR = 0.00005
BATCH_SIZE = 32
embedding_size = 128

save_dir = "final_results"
# pretrained model name
save_path = "lr" + str(LR).replace(".", "") + "b" + str(BATCH_SIZE) + "_3107_50"

NUM_EPOCHS = 100
for LR in [0.00005]:
    # load the pretrained model
    model = ResNet(128).to(device)
    model_path = os.path.join(save_dir, save_path + ".pth")
    model.load_state_dict(torch.load(model_path))

    # Freeze weights
    # for module in model.modules():
    #     if module._get_name() != 'Linear':
    #         for param in module.parameters():
    #             param.requires_grad_(False)
    #     elif module._get_name() == 'Linear':
    #         for param in module.parameters():
    #             param.requires_grad_(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    new_save_path = save_path + "_finetuned_lr" + str(LR).replace(".", "")

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # supervised training
    losses = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        num_items = 0

        for bat in labelled_batch(["FB_13Jan2021_7am_specgenCL0005_ShortPatience", "FB_13Jan2021_8am_specgenCL0005_ShortPatience"], "data/labelled/", "data/Pickle Folder/"):
            optimizer.zero_grad()
            num_items += len(bat['spec1'])
            num_items += len(bat['spec2'])

            combined_embedding1 = torch.empty(0, 128).to(device)
            for ts in bat['spec1']:
                combined_embedding1 = torch.cat((combined_embedding1, model(ts.to(device))), dim=0)
                del ts

            combined_embedding2 = torch.empty(0, 128).to(device)
            for ts in bat['spec2']:
                combined_embedding2 = torch.cat((combined_embedding2, model(ts.to(device))), dim=0)
                del ts

            loss = loss_function(combined_embedding1, combined_embedding2, 0.05)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            del combined_embedding1, combined_embedding2, loss
            torch.cuda.empty_cache()
            gc.collect()

        epoch_loss = running_loss / num_items
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f}")
        losses.append(epoch_loss)

        # save model weights
        torch.save(model.state_dict(), os.path.join(save_dir, new_save_path) + f"_{epoch + 1}.pth")
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizers", new_save_path) + f"_{epoch + 1}.pth")

    # Generate x-axis values (epochs or steps)
    epochs = range(1, len(losses) + 1)

    # Plot the loss graph
    plt.plot(epochs, losses, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Learning rate: {LR}")

    fig_dir = os.path.join(save_dir, "figures")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # save the graph
    plt.savefig(fig_dir + "/" + new_save_path + ".png")
    plt.clf()