import torch
from torch.autograd import Variable
import wandb
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import umap


def Train(criterion, device, label_names, model, optimizer, train_loader, val_loader, epochs, test_loader, isdino=False,
          vis=False):
    iter = 0
    for epoch in range(epochs):
        t0 = time.time()
        print(f'epoch: {epoch}')
        for i, (images, labels) in enumerate(train_loader):
            # if iter == 501:
            #     break
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model.forward(images)
            if not isdino:
                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, labels)
            else:
                loss = outputs
                model.learner.update_moving_average()  # update moving average of teacher encoder and teacher centers
            if vis:
                vis_feature_map_vit(device, epoch, i, iter, model, test_loader)

            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

            iter += 1
            if iter % 50 == 0:
                print(f'iter : {iter}')
                print(loss)
                wandb.log({"loss": loss, "epoch": epoch})
            if iter % 500 == 0:
                pass_model = model.model if isdino else model
                Validation(device, iter, label_names, loss, pass_model, val_loader)
        t1 = time.time()

        time_per_epoch = t1 - t0
        wandb.log({"time_per_epoch": time_per_epoch})
        # save model:
        pass_model = model.model if isdino else model
        torch.save(pass_model.state_dict(), os.path.join(wandb.run.dir, f'model_epoch_{epoch}_.pt'))

        #########################################################################################################
        #                                                 TESTING                                               #
        #########################################################################################################
        print("TESTING TIMZZZ")
        pass_model = model.model if isdino else model
        Testing(device, label_names, pass_model, test_loader)


def vis_feature_map_vit(device, epoch, i, iter, model, test_loader):
    with torch.no_grad():
        if iter % 300 == 0:
            embds = []
            colors = []
            for l, (images2, labels2) in enumerate(test_loader):
                images2 = Variable(images2).to(device)
                labels2 = labels2.to(device)
                # Forward pass only to get logits/output
                outputs2 = model.forward2(images2)

                embds.append(outputs2.view(outputs2.shape[0], -1).cpu().detach().numpy())
                colors.append(labels2.cpu().detach().numpy())
                # print(embds[-1].shape)
                # print(colors[-1].shape)

            embds = np.vstack(embds)
            colors = np.hstack(colors)
            embedding = umap.UMAP(random_state=42).fit_transform(embds)
            plt.scatter(embedding[:, 0], embedding[:, 1], c=colors)
            plt.title(str(i))
            plt.savefig(f"gif_res5/{epoch}__{i}.png")
            plt.show()
            plt.close()


def Validation(device, iter, label_names, loss, model, val_loader):
    # Calculate Accuracy
    correct = 0.0
    correct_arr = [0.0] * 10
    total = 0.0
    total_arr = [0.0] * 10
    # Iterate through test dataset
    with torch.no_grad():
        for images, labels in val_loader:
            images = Variable(images).to(device)
            labels = labels.to(device)
            # Forward pass only to get logits/output
            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)
            correct += (predicted == labels).sum()

            for label in range(4):
                correct_arr[label] += (((predicted == labels) & (labels == label)).sum())
                total_arr[label] += (labels == label).sum()

        accuracy = correct / total

        metrics = {'val accuracy': accuracy}
        for label in range(4):
            metrics['Val Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]

        wandb.log(metrics)

        # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
        # y_true = ground_truth, preds = predictions,
        #                                class_names = class_names)})

        # Print Loss
        print('Iteration: {0} Loss: {1:.2f} Accuracy: {2:.2f}'.format(iter, loss, accuracy))


def Testing(device, label_names, model, test_loader):
    correct = 0.0
    correct_arr = [0.0] * 10
    total = 0.0
    total_arr = [0.0] * 10
    predictions = None
    ground_truth = None
    # Iterate through test dataset
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = Variable(images).to(device)
            labels = labels.to(device)
            # Forward pass only to get logits/output
            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)
            correct += (predicted == labels).sum()

            for label in range(4):
                correct_arr[label] += (((predicted == labels) & (labels == label)).sum())
                total_arr[label] += (labels == label).sum()

            if i == 0:
                predictions = predicted
                ground_truth = labels
            else:
                predictions = torch.cat((predictions, predicted), 0)
                ground_truth = torch.cat((ground_truth, labels), 0)
        accuracy = correct / total

        metrics = {'Test Accuracy': accuracy}
        for label in range(4):
            metrics['Test Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]
        wandb.log(metrics)
        # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
        #                                                    y_true=ground_truth, preds=predictions,
        #                                                    class_names=label_names)})
