"""
ablation of lambda
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101Encoder
import matplotlib.pyplot as plt
from .attention import MultiHeadAttention
from .attention import MultiLayerPerceptron
from .attention import SELayer



class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.criterion_MSE = nn.MSELoss()
        self.alpha = torch.Tensor([1.0, 0.])

        self.iter = 3
        self.fg_sampler = np.random.RandomState(1289)
        self.fg_num = 64  # number of foreground partitions
        self.MHA = MultiHeadAttention(n_head=3, d_model=512, d_k=512, d_v=512)
        self.MLP = MultiLayerPerceptron(dim=512, mlp_dim=1024)
        self.kc_computation = nn.Linear(self.fg_num, 1)
        self.supp_scale = nn.Linear(1024, 512)
        self.spatial_scale = nn.Linear(2, 1)
        self.layer_norm = nn.LayerNorm(512)  # channel dimention = 512

        self.SELayer = SELayer(channel=512)

        self.fusion_1 = nn.Sequential(
            nn.Linear(self.fg_num * 2, self.fg_num, bias=False),
            nn.GELU(),
            nn.Linear(self.fg_num, self.fg_num, bias=False)
        )

        self.fusion_2 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.GELU(),
            nn.Linear(512, 512, bias=False)
        )

        self.query_token = nn.Linear(self.fg_num, self.fg_num)

        # self.postprocess_layer_1 = nn.Linear(64, 32)
        # self.postprocess_layer_2 = nn.Linear(32, 4)

        self.postprocess_layer = nn.Sequential(
            nn.Linear(self.fg_num, 32, bias=False),
            nn.GELU(),
            nn.Linear(32, 16, bias=False)
        )

        self.pred_conv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=False),
            nn.ReLU(),
            # nn.BatchNorm2d(1)
        )

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False, t_loss_scaler=1, n_iters=30):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            qry_mask: query mask
                1 x H x W  tensor
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W
        # supp_mask: (1, 1, 1, 256, 256)

        # Dilate the mask
        kernel = np.ones((10, 10), np.uint8)
        supp_mask_ = supp_mask.cpu().numpy()[0][0][0]
        supp_dilated_mask = cv2.dilate(supp_mask_, kernel, iterations=1)  # (256, 256)
        supp_periphery_mask = supp_dilated_mask - supp_mask_
        supp_periphery_mask = np.reshape(supp_periphery_mask, (supp_bs, self.n_ways, self.n_shots,
                                                               np.shape(supp_periphery_mask)[0],
                                                               np.shape(supp_periphery_mask)[1]))
        supp_dilated_mask = np.reshape(supp_dilated_mask, (supp_bs, self.n_ways, self.n_shots,
                                                           np.shape(supp_dilated_mask)[0],
                                                           np.shape(supp_dilated_mask)[1]))
        supp_periphery_mask = torch.tensor(supp_periphery_mask).cuda()  # (1, 1, 1, 256, 256)  B x Wa x Sh x H x W
        supp_dilated_mask = torch.tensor(supp_dilated_mask).cuda()  # (1, 1, 1, 256, 256)  B x Wa x Sh x H x W

        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        # Get threshold #
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        self.t_ = tao[:self.n_ways * self.n_shots * supp_bs]  # t for support features
        self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]

        # Compute loss #
        align_loss = torch.zeros(1).to(self.device)
        mse_loss = torch.zeros(1).to(self.device)
        loss_qry = torch.zeros(1).to(self.device)
        periphery_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):
            # Extract prototypes #
            # First, object region prototypes #
            supp_fts_ = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                           for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                         range(len(supp_fts))]
            fg_prototypes = [self.getPrototype(supp_fts_[n]) for n in range(len(supp_fts))]  # prototype for support

            # Dilated region prototypes ***************************************************************************** #
            supp_fts_dilated = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_dilated_mask[[epi], way, shot])
                                  for shot in range(self.n_shots)] for way in range(self.n_ways)]
                                for n in range(len(supp_fts))]
            # supp_fts_dilated[0][0][0]: (1, 512)  supp_fts_dilated[1][0][0]: (1, 512)
            fg_prototypes_dilated = [self.getPrototype(supp_fts_dilated[n]) for n in range(len(supp_fts_dilated))]
            # fg_prototypes_dilated[0][0]: (1, 512)  fg_prototypes_dilated[1][0]: (1, 512)
            # Segment periphery region with support images
            supp_pred_object = [
                torch.stack([self.getPred(supp_fts[n][epi][way], fg_prototypes[n][way], self.thresh_pred_[way])
                             for way in range(self.n_ways)], dim=1) for n in range(len(supp_fts))]  # N x Wa x H' x W'
            supp_pred_object = [F.interpolate(supp_pred_object[n], size=img_size, mode='bilinear', align_corners=True)
                                for n in range(len(supp_fts))]
            # supp_pred_object[0]: (1, 1, 256, 256)  supp_pred_object[1]: (1, 1, 256, 256)
            supp_pred_object = [self.alpha[n] * supp_pred_object[n] for n in range(len(qry_fts))]
            supp_pred_object = torch.sum(torch.stack(supp_pred_object, dim=0), dim=0) / torch.sum(self.alpha)
            # supp_pred_object: (1, 1, 256, 256)

            supp_pred_dilated = [
                torch.stack([self.getPred(supp_fts[n][epi][way], fg_prototypes_dilated[n][way], self.thresh_pred_[way])
                             for way in range(self.n_ways)], dim=1) for n in range(len(supp_fts))]  # N x Wa x H' x W'
            supp_pred_dilated = [F.interpolate(supp_pred_dilated[n], size=img_size, mode='bilinear', align_corners=True)
                                 for n in range(len(supp_fts))]
            # supp_pred_dilated[0]: (1, 1, 256, 256)  supp_pred_dilated[1]: (1, 1, 256, 256)
            supp_pred_dilated = [self.alpha[n] * supp_pred_dilated[n] for n in range(len(qry_fts))]
            supp_pred_dilated = torch.sum(torch.stack(supp_pred_dilated, dim=0), dim=0) / torch.sum(self.alpha)
            # Prediction of periphery region
            pred_periphery = supp_pred_dilated - supp_pred_object
            pred_periphery = torch.cat((1.0 - pred_periphery, pred_periphery), dim=1)
            # pred_periphery: (1, 2, 256, 256)  B x C x H x W
            label_periphery = torch.full_like(supp_periphery_mask[epi][0][0], 255, device=supp_periphery_mask.device)
            label_periphery[supp_periphery_mask[epi][0][0] == 1] = 1
            label_periphery[supp_periphery_mask[epi][0][0] == 0] = 0
            # label_periphery: (256, 256)  H x W
            # Compute periphery loss
            eps_ = torch.finfo(torch.float32).eps
            log_prob_ = torch.log(torch.clamp(pred_periphery, eps_, 1 - eps_))
            periphery_loss += self.criterion(log_prob_, label_periphery[None, ...].long()) / self.n_shots / self.n_ways
            # ******************************************************************************************************** #


            # perform a new prediction operation
            fg_partition_prototypes = [[[self.compute_multiple_prototypes(
                self.fg_num, supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot], self.fg_sampler)
                for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in range(len(supp_fts))]

            # QPC module
            qry_pred = [torch.stack(
                [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
            qry_prototype_coarse = [self.getFeatures(qry_fts[n][epi], qry_pred[n][epi]) for n in range(len(qry_fts))]

            # Compute loss use query coarse prototype
            if train:
                qry_pred = [torch.stack(
                    [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
                qry_prototype_coarse = [self.getFeatures(qry_fts[n][epi], qry_pred[n][epi]) for n in
                                        range(len(qry_fts))]

                qry_pred = [self.getPred(qry_fts[n][epi], qry_prototype_coarse[n], self.thresh_pred[epi])
                            for n in range(len(qry_fts))]  # N x Wa x H' x W'
                # qry_pred[0]: (1, 32, 32)  qry_pred[1]: (1, 16, 16)
                qry_pred = [F.interpolate(qry_pred[n][None, ...], size=img_size, mode='bilinear', align_corners=True)
                            for n in range(len(qry_fts))]
                preds = [self.alpha[n] * qry_pred[n] for n in range(len(qry_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
                preds = torch.cat((1.0 - preds, preds), dim=1)

                qry_label = torch.full_like(qry_mask[epi], 255, device=qry_mask.device)
                qry_label[qry_mask[epi] == 1] = 1
                qry_label[qry_mask[epi] == 0] = 0
                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
                loss_qry += self.criterion(log_prob, qry_label[None, ...].long()) / self.n_shots / self.n_ways

            # # The first BATE block
            for i in range(self.iter):
                fg_partition_prototypes = [
                    [[self.BATE(fg_partition_prototypes[n][way][shot][epi], qry_prototype_coarse[n])
                      for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                    range(len(supp_fts))]  # (1, N_f, C)

                supp_proto = [
                    [[torch.mean(fg_partition_prototypes[n][way][shot], dim=1) for shot in range(self.n_shots)]
                     for way in range(self.n_ways)] for n in range(len(supp_fts))]  # (1, C)

                # CQPC module
                qry_pred_coarse = [torch.stack(
                    [self.getPred(qry_fts[n][epi], supp_proto[n][way][epi], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]
                qry_prototype_coarse = [self.getFeatures(qry_fts[n][epi], qry_pred_coarse[n][epi])
                                        for n in range(len(qry_fts))]

            qry_pred_supp = [torch.stack(
                [self.getPred(qry_fts[n][epi], supp_proto[n][way][epi], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
            qry_pred_up_supp = [F.interpolate(qry_pred_supp[n], size=img_size, mode='bilinear', align_corners=True)
                           for n in range(len(qry_fts))]
            pred_supp = [self.alpha[n] * qry_pred_up_supp[n] for n in range(len(qry_fts))]
            preds_supp = torch.sum(torch.stack(pred_supp, dim=0), dim=0) / torch.sum(self.alpha)


            qry_pred_coarse = [torch.stack(
                [self.getPred(qry_fts[n][epi], qry_prototype_coarse[n], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
            qry_pred_up_coarse = [F.interpolate(qry_pred_coarse[n], size=img_size, mode='bilinear', align_corners=True)
                                  for n in range(len(qry_fts))]
            pred_coarse = [self.alpha[n] * qry_pred_up_coarse[n] for n in range(len(qry_fts))]
            preds_coarse = torch.sum(torch.stack(pred_coarse, dim=0), dim=0) / torch.sum(self.alpha)


            t = 1
            preds = t*preds_supp + (1-t)*preds_coarse

            preds = torch.cat((1.0 - preds, preds), dim=1)
            outputs.append(preds)

            # Prototype alignment loss #
            if train:
                align_loss_epi = self.alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                [qry_fts[n][epi] for n in range(len(qry_fts))],
                                                preds, supp_mask[epi])
                align_loss += align_loss_epi

            # Mse alignment loss #
            if train:
                mse_loss_epi = self.proto_alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                    [qry_fts[n][epi] for n in range(len(qry_fts))],
                                                    preds, supp_mask[epi], fg_prototypes)
                mse_loss += mse_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])

        return output, align_loss / supp_bs, mse_loss / supp_bs, periphery_loss / supp_bs, loss_qry / supp_bs

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                # Get predictions
                supp_pred = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred_[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred = [F.interpolate(supp_pred[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]

                # Combine predictions of different feature maps
                preds = [self.alpha[n] * supp_pred[n] for n in range(len(supp_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)

                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))



                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def proto_alignLoss(self, supp_fts, qry_fts, pred, fore_mask, supp_prototypes):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss_sim = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                # Combine prototypes from different scales
                fg_prototypes = [self.alpha[n] * fg_prototypes[n][way] for n in range(len(supp_fts))]
                fg_prototypes = torch.sum(torch.stack(fg_prototypes, dim=0), dim=0) / torch.sum(self.alpha)
                supp_prototypes_ = [self.alpha[n] * supp_prototypes[n][way] for n in range(len(supp_fts))]
                supp_prototypes_ = torch.sum(torch.stack(supp_prototypes_, dim=0), dim=0) / torch.sum(self.alpha)

                # Compute the MSE loss
                loss_sim += self.criterion_MSE(fg_prototypes, supp_prototypes_)

        return loss_sim

    def compute_multiple_prototypes(self, fg_num, sup_fts, sup_fg, sampler):
        """
        Modified from Jian-Wei Zhang

        Parameters
        ----------
        fg_num: int
            Foreground partition numbers
        sup_fts: torch.Tensor
             [B, C, h, w], float32
        sup_fg: torch. Tensor
             [B, h, w], float32 (0,1)
        sampler: np.random.RandomState

        Returns
        -------
        fg_proto: torch.Tensor
            [B, k, C], where k is the number of foreground proxies

        """

        B, C, h, w = sup_fts.shape  # B=1, C=512
        fg_mask = F.interpolate(sup_fg.unsqueeze(0), size=sup_fts.shape[-2:], mode='bilinear')
        fg_mask = fg_mask.squeeze(0).bool()  # [B, h, w] --> bool
        batch_fg_protos = []

        for b in range(B):
            fg_protos = []

            fg_mask_i = fg_mask[b]  # [h, w]

            # Check if zero
            with torch.no_grad():
                if fg_mask_i.sum() < fg_num:
                    fg_mask_i = fg_mask[b].clone()  # don't change original mask
                    fg_mask_i.view(-1)[:fg_num] = True

            # Iteratively select farthest points as centers of foreground local regions
            all_centers = []
            first = True
            pts = torch.stack(torch.where(fg_mask_i), dim=1)
            for _ in range(fg_num):
                if first:
                    i = sampler.choice(pts.shape[0])
                    first = False
                else:
                    dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                    # choose the farthest point
                    i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                pt = pts[i]  # center y, x
                all_centers.append(pt)

            # Assign fg labels for fg pixels
            dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
            fg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

            # Compute fg prototypes
            fg_feats = sup_fts[b].permute(1, 2, 0)[fg_mask_i]  # [N, C]
            for i in range(fg_num):
                proto = fg_feats[fg_labels == i].mean(0)  # [C]
                fg_protos.append(proto)

            fg_protos = torch.stack(fg_protos, dim=1)  # [C, k]
            batch_fg_protos.append(fg_protos)
        fg_proto = torch.stack(batch_fg_protos, dim=0).transpose(1, 2)  # [B, k, C]

        return fg_proto

    def BATE(self, fg_prototypes, qry_prototype_coarse):

        """
        fg_prototypes: (N, 512)
        qry_prototype_coarse: (1, 512)
        """

        # support prototype self-selection


        # S&W module
        qry_prototype_repeat = qry_prototype_coarse.repeat(self.fg_num, 1)
        qry_prototype_repeat = self.query_token(qry_prototype_repeat.t())
        fg_prototypes_new = fg_prototypes + qry_prototype_repeat.t()
        fg_prototypes_mean = torch.mean(fg_prototypes, dim=0).unsqueeze(0)

        # support prototype process in spatical brantch
        spt_prototype_calibration = self.MHA(fg_prototypes_new.unsqueeze(0), fg_prototypes_new.unsqueeze(0),
                                             fg_prototypes_new.unsqueeze(0))
        spt_prototype_calibration = self.MLP(spt_prototype_calibration)  # (1, 64, 512)

        spatial_map = torch.cat(
            (torch.max(spt_prototype_calibration, dim=2)[1].data, torch.mean(spt_prototype_calibration, dim=2)),
            dim=0)  # (2, 64)
        spatial_map = self.spatial_scale(spatial_map.t())
        spatial_activation = F.sigmoid(spatial_map.t())  # (1, 64)
        spatial_activated_prototypes_spatial = torch.mul(spatial_activation.unsqueeze(-1),
                                                         spt_prototype_calibration)  # (1, 64, 512)
        # support prototype process in channel brantch
        # spt_prototype_calibration = self.MHA(fg_prototypes_new.unsqueeze(0), fg_prototypes_new.unsqueeze(0),
        #                                      fg_prototypes_new.unsqueeze(0))
        # spt_prototype_calibration = self.MLP(spt_prototype_calibration)  # (1, 64, 512)
        channel_map = spt_prototype_calibration.permute(0, 2, 1)
        channel_map = channel_map.view(1, 512, 8, 8)  # self.fg_num = 64
        spatial_activated_prototypes_channel = self.SELayer(channel_map).view(1, 512, 64).permute(0, 2,
                                                                                                  1)  # (1, 64, 512)

        # prototypes fusion from different angles
        spt_prototype_fusion_1 = torch.cat((spatial_activated_prototypes_spatial, spt_prototype_calibration),
                                           dim=1).squeeze(
            0)  # (128, 512)
        spt_prototype_fusion_1 = self.fusion_1(spt_prototype_fusion_1.t()).t()  # (64, 512)
        spt_prototype_fusion_1 = self.MHA(spt_prototype_fusion_1.unsqueeze(0), spt_prototype_fusion_1.unsqueeze(0),
                                          spt_prototype_fusion_1.unsqueeze(0))
        spt_prototype_fusion_1 = self.MLP(spt_prototype_fusion_1)

        # *** #
        spt_prototye_fusion_2 = torch.cat((spatial_activated_prototypes_channel, spt_prototype_calibration),
                                          dim=2).squeeze(
            0)  # (64, 1024)
        spt_prototye_fusion_2 = self.fusion_2(spt_prototye_fusion_2)  # (64, 512)
        spt_prototye_fusion_2 = self.MHA(spt_prototye_fusion_2.unsqueeze(0), spt_prototye_fusion_2.unsqueeze(0),
                                         spt_prototye_fusion_2.unsqueeze(0))
        spt_prototye_fusion_2 = self.MLP(spt_prototye_fusion_2)

        # S&W module
        spt_prototype_calibration = self.MHA(fg_prototypes_new.unsqueeze(0), fg_prototypes_new.unsqueeze(0),
                                             fg_prototypes_new.unsqueeze(0))
        spt_prototype_calibration = self.MLP(spt_prototype_calibration)
        qry_prototype_calibration = self.MHA(qry_prototype_coarse.unsqueeze(0), qry_prototype_coarse.unsqueeze(0),
                                             qry_prototype_coarse.unsqueeze(0))
        qry_prototype_calibration = self.MLP(qry_prototype_calibration)
        A = torch.mm(spt_prototype_calibration.squeeze(0), qry_prototype_calibration.squeeze(0).t())

        # kc computation
        # A_ = A.t().squeeze(0).squeeze(0)
        # kc = self.kc_computation(A_)
        # print(kc, A.min(), A.mean(), A.max())
        kc = ((A.min() + A.mean())/2).floor()
        if A is not None:
            S = torch.zeros(A.size(), dtype=torch.float).cuda()
            S[A < kc] = -10000.0

        A_1 = torch.softmax((A + S), dim=0)

        # fg_prototypes_scaled = self.supp_scale(fg_prototypes.t())

        A = torch.mm(A_1, fg_prototypes_mean)

        # A = self.supp_scale(torch.cat((A, fg_prototypes), dim=1))  # (64, 512)
        # fg_prototypes_new = A_2 * fg_prototypes_new
        A = self.layer_norm(A + fg_prototypes_new)

        # rest Transformer operation
        T = self.MHA(A.unsqueeze(0), fg_prototypes_new.unsqueeze(0), fg_prototypes_new.unsqueeze(0))
        T = self.MLP(T)

        spt_prototye_fusion = self.supp_scale(
            torch.cat((spt_prototype_fusion_1.squeeze(0), spt_prototye_fusion_2.squeeze(0)), dim=1))

        T = self.supp_scale(torch.cat((T.squeeze(0), spt_prototye_fusion), dim=1))
        # T = T.squeeze(0) + spt_prototye_fusion
        T = self.MHA(T.unsqueeze(0), T.unsqueeze(0), T.unsqueeze(0))
        T = self.MLP(T)

        T = self.layer_norm(T + fg_prototypes.unsqueeze(0))


        return T




