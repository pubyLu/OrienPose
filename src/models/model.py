import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from src.models.fuse_model import OrienCrossAttnBlock as Mixer
from src.models.orient.vision_tower import DINOv2_MLP
import torch.nn as nn

def unnormalize_to_zero_to_one(t):
    img = t.clone()
    img = (img + 1) * 0.5
    return img.clamp(0, 1)


class PoseConditional(pl.LightningModule):
    def __init__(
            self,
            u_net,
    ):
        super().__init__()
        self.u_net = u_net
        self.orient_evaluator = self._load_orient_model()
        # 占位Mixer
        self.mixer = Mixer()
    def _load_orient_model(self, pretrained="/project_root/pretrained/dino.pt"):
        orient = DINOv2_MLP(dino_mode='large',
                            in_dim=1024,
                            out_dim=360 + 180 + 180 + 2,
                            evaluate=True,
                            mask_dino=False,
                            frozen_back=False
                            )
        orient.load_state_dict(torch.load(pretrained))
        orient.eval()
        return orient


    def forward(self, query, ref, relR):
        '''
        query: target view
        ref: reference view
        relR: relative pose from ref to query.
        '''
        orient_query = self.orient_evaluator({'pixel_values': query})[:, :720]
        orient_ref = self.orient_evaluator({'pixel_values': ref})[:, :720]
        query_feat = self.u_net.encoder.encode_image(query)
        ref_feat = self.u_net.encoder.encode_image(ref, mode="mode")
        orient_feat_ref = self.mixer(ref_feat, orient_ref)
        ref_feat = orient_feat_ref
        pred_query_feat = self.u_net(ref_feat, relR)
        pred_rgb = self.u_net.encoder.decode_latent(pred_query_feat)
        pred_rgb = unnormalize_to_zero_to_one(pred_rgb)
        return pred_rgb, pred_query_feat, orient_query, query_feat

    def my_generate_templates(self, ref, relRs):
        '''
        参考图像+ 参考与模型之间的相对姿态--> 生成图像+生成图像编码
        :param self:
        :param ref:
        :param relRs:
        :param gt_templates:
        :param visualize:
        :return:
        '''
        b, c, h, w = ref.shape
        num_templates = relRs.shape[1]
        # keep all predicted features of template for retrieval later
        if hasattr(self.u_net.encoder, "decode_latent") and callable(
                self.u_net.encoder.decode_latent
        ):
            pred = torch.zeros((b, num_templates, c, h, w), device=ref.device)
        else:
            pred = None
        pred_feat = torch.zeros(
            (b, num_templates, self.u_net.encoder.latent_dim, int(h / 8), int(w / 8)),
            device=ref.device,
        )
        reference_feat = self.u_net.encoder.encode_image(ref, mode="mode")
        orient_ref = self.orient_evaluator({"pixel_values":ref})[:, :720]
        reference_feat = self.mixer(reference_feat, orient_ref)
        for idx in range(0, num_templates):
            # get output of sample
            pred_feat_i, pred_rgb_i = self.my_sample_for_generate_templates(ref=reference_feat, relR=relRs[:, idx, :]) # 这里是N个refs一起去做的
            pred_feat[:, idx] = pred_feat_i

            if pred_rgb_i is not None:
                pred[:, idx] = pred_rgb_i

        return pred_feat, pred

    def my_retrieval(self, query, template_feat, template_gen_rgb):
        '''
        query:[b, c, h, w]
        template_feat:[b, n, c, h, w]
        template_gen_rgb:[b, n, c, h, w]

        '''
        bs = query.shape[0]
        oa_distances = []
        for i in range(bs):
            qry_orient = self.orient_evaluator({"pixel_values":query[i:i+1]})[:, :720] # [1, 720]
            template_orient = self.orient_evaluator({"pixel_values":template_gen_rgb[i]})[:, :720] # [n, 720]
            logq_pred = F.log_softmax(template_orient, dim=-1) # [n, 720]
            logp_tgt = F.log_softmax(qry_orient, dim=-1).repeat(logq_pred.shape[0], 1) # [n, 720]

            kl_azimuth = F.kl_div(
                input=logq_pred[:, :360],  # log q
                target=logp_tgt[:, :360],  # log p
                reduction='none',
                log_target=True
            ).sum(dim=-1)  # 对 720 维求和 -> [n]

            kl_polar = F.kl_div(
                input=logq_pred[:, 360:540],  # log q
                target=logp_tgt[:, 360:540],  # log p
                reduction='none',
                log_target=True
            ).sum(dim=-1)  # 对 720 维求和 -> [n]

            kl_rotation = F.kl_div(
                input=logq_pred[:, 540:],  # log q
                target=logp_tgt[:, 540:],  # log p
                reduction='none',
                log_target=True
            ).sum(dim=-1)  # 对 720 维求和 -> [n]
            kl_per_template = kl_azimuth + kl_polar+kl_rotation
            oa_distances.append(kl_per_template)
        oa_distances = - torch.stack(oa_distances, dim=0)

        num_templates = template_feat.shape[1]
        if self.testing_config.similarity_metric == "l2":
            query_feat = self.u_net.encoder.encode_image(query, mode="mode")
            query_feat = query_feat.unsqueeze(1).repeat(1, num_templates, 1, 1, 1)

            distance = (query_feat - template_feat) ** 2
            distance = torch.norm(distance, dim=2)
            similarity = -distance.sum(axis=3).sum(axis=2)  # B x N
            new_similarity = similarity + oa_distances
            _, nearest_idx = new_similarity.topk(k=5, dim=1)
            return nearest_idx

    def myTest_step(
            self, batch
    ):
        # visualize same loss as training
        query = batch["query"]
        ref = batch["ref"]
        template_relRs = batch["template_relRs"]

        pred_feat, pred_rgb = self.my_generate_templates(
            ref=ref,
            relRs=template_relRs,
        )

        nearest_idx = self.my_retrieval(query=query, template_feat=pred_feat, template_gen_rgb=pred_rgb).cpu().numpy()

        return nearest_idx
