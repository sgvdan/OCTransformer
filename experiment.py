import numpy as np
import wandb

import util
from analysis.stats import get_binary_prediction
from analysis.visualizer import plot_attention, plot_masks, plot_slices, plot_gradcam, get_masks, get_gradcam, \
    low_dimension_plot
from data.boe_chiu_data import BOEChiuDataset, get_boe_chiu_transform
from data.hadassah_data import setup_hadassah
from data.hadassah_mix import MixedDataset
from data.kermany_data import setup_kermany
from models.bank import ModelsBank
from config import default_config
from logger import Logger
from train.train import Trainer
import torch


class Experiment:
    def __init__(self, config):
        assert config is not None
        self.config = config

        util.make_deterministic()

        # Initiate W&B
        if self.config.log:
            wandb.login()
            wandb.init(project=self.config.project, group=self.config.log_group, config=self.config)
            self.config = wandb.config

        # Set up Logger
        self.logger = Logger(self.config)

        # Set up Data
        self.train_loader, self.eval_loader, self.test_loader = self.setup_data()

        # Set up Model Bank
        self.model_bank = ModelsBank(self.config)

        # Set up Trainer
        self.trainer = Trainer(self.config, self.train_loader, self.eval_loader,
                               self.test_loader, self.model_bank, self.logger)

        # Set up Model Environment
        self.model, self.optimizer, self.scheduler = self.model_bank.get_environment()
        self.criterion = self.model_bank.get_balanced_criterion(self.train_loader)

    def setup_data(self):
        if self.config.dataset == 'hadassah':
            return setup_hadassah(self.config)
        elif self.config.dataset == 'kermany':
            return setup_kermany(self.config)
        else:
            raise NotImplementedError

    def train(self):
        self.trainer.train(self.model, self.criterion, self.optimizer, self.scheduler, self.config.epochs)

        if self.config.load_best_model:
            self.model_bank.load_best(self.model, self.optimizer, self.scheduler)  # Refresh model (avoid over fitting)

    def test(self):
        return self.trainer.test(self.model)

    def visualize(self):
        # mix_dataset = MixedDataset(self.test_loader.dataset)
        # mix_loader = torch.utils.data.DataLoader(dataset=mix_dataset, batch_size=self.config.batch_size)

        shuffle_test = torch.utils.data.DataLoader(dataset=self.test_loader.dataset,
                                                   batch_size=self.config.batch_size,
                                                   shuffle=True)
        count = 0
        for idx, (volume, label) in enumerate(shuffle_test):
            if count > 5:
                break

            # Generate Weighted GradCam Masks per each positive label
            pred, _ = self.trainer._feed_forward(self.model, volume, label, mode='eval')
            binary_pred = get_binary_prediction(pred.cpu(), torch.tensor(self.model_bank.bank_record[self.model.name]['thresholds']))
            target_labels = binary_pred.nonzero()[:, 1].tolist()  # gather all positive labels

            if not target_labels:
                continue

            count += 1

            # Keep slices
            plot_slices(volume.squeeze(dim=0), logger=self.logger, title='raw')

            # Keep Attention Maps
            attn = self.model.get_attention_map(volume)
            plot_attention(attn, logger=self.logger, title='attention')

            # Keep GradCAM
            cam = get_gradcam(input_tensor=volume, model=self.model.model.patch_embed,
                              target_layers=[self.model.model.patch_embed.backbone.layer4[-1]],
                              type=self.config.gradcam_type, device=self.config.device,
                              aug_smooth=self.config.aug_smooth, eigen_smooth=self.config.eigen_smooth)
            plot_gradcam(volume.squeeze(dim=0), cam, logger=self.logger, title='gradcam')

            # Keep Masks
            mask = get_masks(attn, cam, std_thresh=self.config.mask_std_thresh)
            plot_masks(volume.squeeze(dim=0), mask, logger=self.logger, title='mask')

            self.logger.flush_images(name='vis-' + str(idx))
            print("Model's {} prediction:".format(idx), [self.config.labels[idx] for idx in target_labels])

    def boe_chiu_eval(self):
        assert self.config.num_slices == 11  # Need to comply with BOE_Chiu's slices count
        boe_chiu_dataset = BOEChiuDataset('/home/projects/ronen/sgvdan/workspace/datasets/2015_BOE_Chiu/mat_dataset/train',
                                          transformations=get_boe_chiu_transform(self.config.input_size))
        boe_chiu_loader = torch.utils.data.DataLoader(dataset=boe_chiu_dataset,
                                                      batch_size=self.config.batch_size,
                                                      shuffle=True)
        dsc = []
        test_table = wandb.Table(columns=['id', 'prediction', 'dsc'])
        for idx, (volume, label) in enumerate(boe_chiu_loader):
            # Generate Weighted GradCam Masks per each positive label
            pred, _ = self.trainer._feed_forward(self.model, volume, label, mode='eval')
            binary_pred = get_binary_prediction(pred.cpu(), torch.tensor(self.model_bank.bank_record[self.model.name]['thresholds']))
            target_labels = binary_pred.nonzero()[:, 1].tolist()  # gather all positive labels

            if not target_labels:
                continue

            # Keep slices
            plot_slices(volume.squeeze(dim=0), logger=self.logger, title='raw')

            # Keep Attention Maps
            attn = self.model.get_attention_map(volume)
            plot_attention(attn, logger=self.logger, title='attention')

            # Keep GradCAM
            cam = get_gradcam(input_tensor=volume, model=self.model.model.patch_embed,
                              target_layers=[self.model.model.patch_embed.backbone.layer4[-1]],
                              type=self.config.gradcam_type, device=self.config.device,
                              aug_smooth=self.config.aug_smooth, eigen_smooth=self.config.eigen_smooth)
            plot_gradcam(volume.squeeze(dim=0), cam, logger=self.logger, title='gradcam')

            # Keep Masks
            mask = get_masks(attn, cam, std_thresh=self.config.mask_std_thresh)

            fluid_gt = (label.squeeze() == 9).cpu().numpy().astype(float)
            roi_pred = (mask > 0).astype(float)

            dsc.append(2 * np.multiply(fluid_gt, roi_pred).sum() / (fluid_gt + roi_pred).sum())
            print("DSC: ", dsc[-1])

            plot_masks(volume.squeeze(dim=0), mask, logger=self.logger, title='mask')

            y = np.multiply(fluid_gt, roi_pred)
            plot_masks(volume.squeeze(dim=0), y, logger=self.logger, title='intersection')

            y = 0.5 * (fluid_gt + roi_pred)
            plot_masks(volume.squeeze(dim=0), y, logger=self.logger, title='union')

            self.logger.flush_images(name='boe-chiu-' + str(idx))

            str_pred = str([self.config.labels[idx] for idx in target_labels])
            print("Model's {} prediction:".format(idx), str_pred)
            test_table.add_data('boe-chiu-' + str(idx), str_pred, dsc[-1])

        mean_dsc = np.array(dsc).mean()
        test_table.add_data('boe-chiu-mean', '-', mean_dsc)

        self.logger.log({'DSC': test_table})
        self.logger.log({'MEAN_DSC': mean_dsc})
        print('MEAN DSC:', mean_dsc)


def main():
    experiment = Experiment(default_config)
    experiment.train()
    backbone = experiment.model.model.patch_embed
    y = torch.concat([torch.stack(experiment.logger.train_gt), torch.stack(experiment.logger.eval_gt)])
    z = torch.argmax(y, dim=1)
    data = experiment.model.accum_cls_token
    low_dimension_plot(data, z, "ViT T-SNE VOLUME TRAINING+EVALUATION (DR) projection")

    # y = y[:, 0]  # LOOK ONLY AT DR
    # y = y.unsqueeze(dim=1).expand(-1, experiment.config.num_slices)
    #
    # num_slices, embedding_dim = experiment.config.num_slices, experiment.config.embedding_dim
    # num_samples = y.shape[0]
    # y = y.reshape(num_samples * num_slices)

    # data = backbone.accum_resnet_tokens.reshape(num_samples * num_slices, embedding_dim)
    # low_dimension_plot(data, y, "ResNet T-SNE TRAINING+EVALUATION (DR) projection")
    #
    # data = backbone.accum_gist_tokens.reshape(num_samples * num_slices, 64)
    # low_dimension_plot(data, y, "MGU-Net T-SNE TRAINING+EVALUATION (DR) projection")
    #
    # data = backbone.accum_concat_tokens.reshape(num_samples * num_slices, embedding_dim + 64)
    # low_dimension_plot(data, y, "Concat T-SNE TRAINING+EVALUATION (DR) projection")

    experiment.test()

    y = torch.argmax(torch.stack(experiment.logger.test_gt), dim=1)
    data = experiment.model.accum_cls_token
    low_dimension_plot(data, y, "ViT T-SNE VOLUME TEST (DR) projection")
    # experiment.logger.log_curves()
    # experiment.visualize()
    # experiment.boe_chiu_eval()


if __name__ == '__main__':
    main()
