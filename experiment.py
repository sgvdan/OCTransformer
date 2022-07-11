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


class A:
    def __init__(self, dataset):
        self.dataset = dataset
        A.volume_path = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        A.volume_path = self.dataset.samples[idx].volume_path
        return self.dataset[idx]


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
        score = self.trainer.test(self.model)
        self.logger.log_summary('overall_score', score)
        return score

    def visualize(self):
        # mix_dataset = MixedDataset(self.test_loader.dataset)
        # mix_loader = torch.utils.data.DataLoader(dataset=mix_dataset, batch_size=self.config.batch_size)

        shuffle_test = torch.utils.data.DataLoader(dataset=A(self.test_loader.dataset),
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
            print("{} path:".format(idx), A.volume_path)

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

    def attention_eval(self):
        mix_dataset = MixedDataset(self.test_loader.dataset, count=1000, slices_num=self.config.num_slices)
        mix_loader = torch.utils.data.DataLoader(dataset=mix_dataset, batch_size=self.config.batch_size)

        score = []
        for idx, ((volume, sick_idx), _) in enumerate(mix_loader):
            attn = self.model.get_attention_map(volume).squeeze(axis=0)
            sick_idx = sick_idx.squeeze(axis=0)
            healthy_avg = np.average(np.delete(attn, sick_idx))
            scr = (attn[sick_idx] > healthy_avg).astype(float).sum() / len(sick_idx)  # How many SICK slices are above healthy avg?

            score.append(scr)
            print('{} score: {}'.format(idx, scr))

        print('Average score: ', np.mean(score))

        return score


def main():
    experiment = Experiment(default_config)
    experiment.train()
    experiment.test()
    experiment.logger.log_curves()
    # experiment.attention_eval()
    # experiment.visualize()
    experiment.boe_chiu_eval()


if __name__ == '__main__':
    main()
