import torch
from torch.autograd import Variable
import wandb
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import umap

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import scipy.interpolate
from matplotlib import animation


def axis_bounds(embedding):
    left, right = embedding.T[0].min(), embedding.T[0].max()
    bottom, top = embedding.T[1].min(), embedding.T[1].max()
    adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
    return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]


def Train(criterion, device, label_names, model, optimizer, train_loader, val_loader, epochs, test_loader, isdino=False,
          vis=False, isvit=False, isnext=False):
    aligned_mapper = None
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
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

            if vis:
                if iter % 50000 == 0:
                    if isvit:
                        embds, colors, aligned_mapper = vis_feature_map_vit(device, epoch, i, iter, model, test_loader,
                                                                            aligned_mapper)
                    elif isnext:
                        embds, colors, aligned_mapper = vis_feature_map_next(device, epoch, i, iter, model, test_loader,
                                                                             aligned_mapper)
            iter += 1
            if iter % 50 == 0:
                print(f'iter : {iter}')
                print(loss)
                wandb.log({"loss": loss, "epoch": epoch})
            if iter % 500 == 0 and not vis:
                pass_model = model.model if isdino else model
                Validation(device, iter, label_names, loss, pass_model, val_loader)
        t1 = time.time()

        time_per_epoch = t1 - t0
        wandb.log({"time_per_epoch": time_per_epoch})
        # save model:
        #########################################################################################################
        #                                                 TESTING                                               #
        #########################################################################################################
        if not vis or epoch == epochs:
            print("TESTING TIMZZZ ")
            pass_model = model.model if isdino else model
            Testing(device, label_names, pass_model, test_loader)
    pass_model = model.model if isdino else model
    torch.save(pass_model.state_dict(), os.path.join(wandb.run.dir, f'model.pt'))

    if vis:
        vis_gif(aligned_mapper, colors, embds)


def vis_gif(aligned_mapper, colors, embds):
    n_embeddings = len(aligned_mapper.embeddings_)
    es = aligned_mapper.embeddings_
    embedding_df = pd.DataFrame(np.vstack(es), columns=('x', 'y'))
    embedding_df['z'] = np.repeat(np.linspace(0, 1.0, n_embeddings), es[0].shape[0])
    embedding_df['id'] = np.tile(np.arange(es[0].shape[0]), n_embeddings)
    embedding_df['digit'] = np.tile(colors, n_embeddings)
    fx = scipy.interpolate.interp1d(
        embedding_df.z[embedding_df.id == 0], embedding_df.x.values.reshape(n_embeddings, embds.shape[0]).T,
        kind="cubic"
    )
    fy = scipy.interpolate.interp1d(
        embedding_df.z[embedding_df.id == 0], embedding_df.y.values.reshape(n_embeddings, embds.shape[0]).T,
        kind="cubic"
    )
    z = np.linspace(0, 1.0, 100)
    # palette = px.colors.diverging.Spectral
    interpolated_traces = [fx(z), fy(z)]
    # traces = [
    #     go.Scatter3d(
    #         x=interpolated_traces[0][i],
    #         y=interpolated_traces[1][i],
    #         z=z * 3.0,
    #         mode="lines",
    #         line=dict(
    #             color=palette[colors[i]],
    #             width=3.0
    #         ),
    #         opacity=1.0,
    #     )
    #     for i in range(embds.shape[0])
    # ]
    # fig = go.Figure(data=traces)
    # fig.update_layout(
    #     width=800,
    #     height=700,
    #     autosize=False,
    #     showlegend=False,
    # )
    # fig.show()
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax_bound = axis_bounds(np.vstack(aligned_mapper.embeddings_))
    scat = ax.scatter([], [], s=2)
    scat.set_array(colors)
    scat.set_cmap('Spectral')
    text = ax.text(ax_bound[0] + 0.5, ax_bound[2] + 0.5, '')
    ax.axis(ax_bound)
    ax.set(xticks=[], yticks=[])
    plt.tight_layout()
    offsets = np.array(interpolated_traces).T
    num_frames = offsets.shape[0]

    def animate(i):
        scat.set_offsets(offsets[i])
        text.set_text(f'Frame {i}')
        return scat

    anim = animation.FuncAnimation(
        fig,
        init_func=None,
        func=animate,
        frames=num_frames,
        interval=40)
    anim.save("aligned_umap__anim__9__5_3__8__1.gif", writer="pillow")
    plt.close(anim._fig)


def vis_feature_map_vit(device, epoch, i, iter, model, test_loader, aligned_mapper):
    with torch.no_grad():
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
        if i == 0 and epoch == 0:
            relation_dict = {i: i for i in range(embds.shape[0])}
            relation_dicts = [relation_dict.copy() for i in range(2 - 1)]
            aligned_mapper = umap.AlignedUMAP().fit([embds, embds], relations=relation_dicts)
        else:
            umap_viz(embds, aligned_mapper)
        # embedding = umap.UMAP(random_state=42).fit_transform(embds)
        # plt.scatter(embedding[:, 0], embedding[:, 1], c=colors)
        # plt.title(str(i))
        # plt.savefig(f"gif_res5/{epoch}__{i}.png")
        # plt.show()
        # plt.close()
    return embds, colors, aligned_mapper


def vis_feature_map_next(device, epoch, i, iter, model, test_loader, aligned_mapper):
    with torch.no_grad():
        embds = []
        colors = []
        for l, (images2, labels2) in enumerate(test_loader):
            images2 = Variable(images2).to(device)
            labels2 = labels2.to(device)
            # Forward pass only to get logits/output
            outputs2 = model.forward_features(images2)

            embds.append(outputs2.view(outputs2.shape[0], -1).cpu().detach().numpy())
            colors.append(labels2.cpu().detach().numpy())
            # print(embds[-1].shape)
            # print(colors[-1].shape)

        embds = np.vstack(embds)
        colors = np.hstack(colors)
        if i == 0 and epoch == 0:
            relation_dict = {i: i for i in range(embds.shape[0])}
            relation_dicts = [relation_dict.copy() for i in range(2 - 1)]
            aligned_mapper = umap.AlignedUMAP().fit([embds, embds], relations=relation_dicts)
        else:
            umap_viz(embds, aligned_mapper)
        if False:
            embedding = umap.UMAP(random_state=42, n_components=3).fit_transform(embds)
            point_cloud = np.hstack([embedding, colors.reshape(-1, 1)])
            wandb.log({f"3D_UMAP_FeatureMap_": wandb.Object3D(point_cloud)})
        # plt.scatter(embedding[:, 0], embedding[:, 1], c=colors)
        # plt.title(str(i))
        # plt.savefig(f"gif_res5/{epoch}__{i}.png")
        # plt.show()
        # plt.close()
    return embds, colors, aligned_mapper


def umap_viz(embds, aligned_mapper):
    relation_dict = {i: i for i in range(embds.shape[0])}
    relation_dicts = [relation_dict.copy() for _ in range(embds.shape[0] - 1)]
    aligned_mapper.update(embds, relations={v: k for k, v in relation_dicts[0].items()})


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
