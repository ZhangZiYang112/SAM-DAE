import argparse
import torch
import os
import pdb
import sys
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from tqdm import tqdm as tqdm_load
from pancreas_utils import *
from test_util import *
from losses import *
from dataloaders import get_ema_model_and_dataloader
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from semisam_branch import semisam_branch, get_sam_branch
from networks.VNetDAE import VNetDAE, LightweightDAE
from utils import ramps


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--data_root', type=str, default='../Datasets/pancreas/data', help='Pancreas data root')
parser.add_argument('--split_name', type=str, default='pancreas', help='Dataset split name')
parser.add_argument('--result_dir', type=str, default='result/pancreas/', help='Result directory')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--pretraining_epochs', type=int, default=101, help='Pre-training epochs')
parser.add_argument('--self_training_epochs', type=int, default=321, help='Self-training epochs')
parser.add_argument('--label_percent', type=int, default=20, help='Labeled data percentage, 10 or 20')
parser.add_argument('--seed', type=int, default=2020, help='Random seed')
parser.add_argument('--skip_pretrain', type=int, default=0, help='Skip segmentation pre-training')
parser.add_argument('--use_sam', type=int, default=1, help='Whether to use SAM-Med3D enhancement')
parser.add_argument('--sam_prompt', type=str, default='unc', help='SAM prompt strategy: unc, centroid, random')
parser.add_argument('--sam_weight', type=float, default=0.1, help='Weight for SAM consistency loss')
parser.add_argument('--sam_rampup', type=float, default=40.0, help='Rampup for SAM loss weight')
parser.add_argument('--sam_skip_on_error', type=int, default=0, help='Skip SAM loss when SAM-Med3D is unavailable or errors')
parser.add_argument('--use_dae', type=int, default=1, help='Whether to use DAE for pseudo-label denoising')
parser.add_argument('--dae_model_path', type=str, default='', help='Path to pre-trained DAE model')
parser.add_argument('--dae_gamma', type=float, default=1.0, help='Uncertainty weight factor for DAE')
parser.add_argument('--dae_weight', type=float, default=0.5, help='Weight for DAE consistency loss')
parser.add_argument('--dae_rampup', type=float, default=40.0, help='Rampup epochs for DAE loss weight')
parser.add_argument('--dae_lightweight', type=int, default=0, help='Use lightweight DAE model')
parser.add_argument('--dae_pretrain_epochs', type=int, default=100, help='Epochs for DAE pre-training')
parser.add_argument('--dae_pretrain_lr', type=float, default=1e-3, help='Learning rate for DAE pre-training')
parser.add_argument('--dae_noise_ratio', type=float, default=0.2, help='Noise ratio for DAE training')
parser.add_argument('--skip_dae_pretrain', type=int, default=0, help='Skip DAE pre-training')
args = parser.parse_args()

"""Global Variables"""
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
seed_test = args.seed
seed_reproducer(seed = seed_test)

data_root, split_name = args.data_root, args.split_name
result_dir = args.result_dir
mkdir(result_dir)
batch_size, lr = args.batch_size, args.lr
pretraining_epochs, self_training_epochs = args.pretraining_epochs, args.self_training_epochs
pretrain_save_step, st_save_step, pred_step = 10, 20, 5
alpha, consistency, consistency_rampup = 0.99, 0.1, 40
label_percent = args.label_percent
u_weight = 1.5
connect_mode = 2
try_second = 1
sec_t = 0.5
self_train_name = 'self_train'

sub_batch = int(batch_size/2)
consistency_criterion = softmax_mse_loss
CE = nn.CrossEntropyLoss()
CE_r = nn.CrossEntropyLoss(reduction='none')
DICE = DiceLoss(nclass=2)
patch_size = 64
volume_size = (96, 96, 96)

logger = None


def cmp_dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def create_DAE(lightweight=False, trainable=False):
    """Create a DAE that denoises one-channel foreground probability/label maps."""
    if lightweight:
        net = LightweightDAE(n_channels=1, n_classes=1, n_filters=8, normalization='instancenorm')
    else:
        net = VNetDAE(
            n_channels=1,
            n_classes=1,
            n_filters=16,
            normalization='instancenorm',
            has_dropout=False,
            input_size=volume_size,
            is_LS_noise=True,
            emb_dim=256,
        )
    model = nn.DataParallel(net).cuda()
    if not trainable:
        for param in model.parameters():
            param.detach_()
    return model


def add_noise_to_label(label, noise_ratio=0.2):
    noise = torch.rand_like(label.float())
    flip_mask = noise < noise_ratio
    noisy_label = label.float().clone()
    noisy_label[flip_mask] = 1.0 - noisy_label[flip_mask]
    gaussian_noise = torch.randn_like(noisy_label) * 0.1
    return torch.clamp(noisy_label + gaussian_noise, 0.0, 1.0)


def get_sam_consistency_weight(epoch, max_epoch):
    if args.sam_rampup <= 0:
        return args.sam_weight
    current = np.clip(epoch - 1, 0.0, args.sam_rampup)
    return args.sam_weight * ramps.cosine_rampdown(current, args.sam_rampup)


def get_dae_consistency_weight(epoch):
    return args.dae_weight * ramps.sigmoid_rampup(epoch, args.dae_rampup)


def pretrain_dae(lab_loader, snapshot_path):
    """Pre-train DAE on labeled pancreas masks to learn a label-space prior."""
    snapshot_path.mkdir(exist_ok=True)
    dae_model = create_DAE(lightweight=args.dae_lightweight, trainable=True)
    optimizer_dae = optim.Adam(dae_model.parameters(), lr=args.dae_pretrain_lr)
    scheduler_dae = optim.lr_scheduler.CosineAnnealingLR(optimizer_dae, args.dae_pretrain_epochs)
    bce_loss = nn.BCEWithLogitsLoss()

    dae_model.train()
    best_loss = float('inf')
    logger.info("DAE pre-training, save path: {}".format(str(snapshot_path)))
    logger.info("DAE epochs: {}, noise ratio: {}".format(args.dae_pretrain_epochs, args.dae_noise_ratio))

    for epoch in tqdm_load(range(1, args.dae_pretrain_epochs + 1), ncols=70, desc="DAE Pre-train"):
        epoch_loss = 0.0
        epoch_dice = 0.0
        num_batches = 0

        for _, lab in lab_loader:
            lab = lab.cuda()
            if lab.dim() == 4:
                lab = lab.unsqueeze(1).float()
            else:
                lab = lab.float()

            noisy_lab = add_noise_to_label(lab, noise_ratio=args.dae_noise_ratio)
            dae_output, _ = dae_model(noisy_lab)
            loss_bce = bce_loss(dae_output, lab)
            dae_pred = torch.sigmoid(dae_output)
            loss_dice = cmp_dice_loss(dae_pred, lab)
            loss = loss_bce + loss_dice

            optimizer_dae.zero_grad()
            loss.backward()
            optimizer_dae.step()

            epoch_loss += loss.item()
            epoch_dice += 1 - loss_dice.item()
            num_batches += 1

        scheduler_dae.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_dice = epoch_dice / max(num_batches, 1)

        if epoch % 10 == 0:
            logger.info(
                'DAE Epoch {}: loss={:.4f}, dice={:.4f}, lr={:.6f}'.format(
                    epoch, avg_loss, avg_dice, scheduler_dae.get_last_lr()[0]
                )
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(dae_model.state_dict(), str(snapshot_path / 'best_dae_model.pth'))
            logger.info("Save best DAE model, loss={:.4f}".format(avg_loss))

    final_path = snapshot_path / 'final_dae_model.pth'
    torch.save(dae_model.state_dict(), str(final_path))
    logger.info("DAE pre-training finished. Final model saved to {}".format(str(final_path)))
    return snapshot_path / 'best_dae_model.pth'

def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def pretrain(net1, net2, optimizer1, optimizer2, lab_loader_a, lab_loader_b, test_loader):
    """pretrain image- & patch-aware network"""

    """Create Path"""
    save_path = Path(result_dir) / 'pretrain'
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = cutmix_config_log(save_path, tensorboard=True)
    logger.info("cutmix Pretrain, patch_size: {}, save path: {}".format(patch_size, str(save_path)))

    max_dice1 = 0
    max_dice2 = 0
    measures = CutPreMeasures(writer, logger)

    for epoch in tqdm_load(range(1, pretraining_epochs + 1), ncols=70):
        measures.reset()
        """Testing"""
        if epoch % 5 == 0:
            net1.eval()
            net2.eval()
            avg_metric1, _ = test_calculate_metric(net1, test_loader.dataset, s_xy=16, s_z=4)
            avg_metric2, _ = test_calculate_metric(net2, test_loader.dataset, s_xy=16, s_z=4)

            logger.info('average metric is : {}'.format(avg_metric1))
            logger.info('average metric is : {}'.format(avg_metric2))
            val_dice1 = avg_metric1[0]
            val_dice2 = avg_metric2[0]

            if val_dice1 > max_dice1:
                save_net_opt(net1, optimizer1, save_path / f'best_ema{label_percent}_pre_vnet.pth', epoch)
                max_dice1 = val_dice1

            if val_dice2 > max_dice2:
                save_net_opt(net2, optimizer2, save_path / f'best_ema{label_percent}_pre_resnet.pth', epoch)
                max_dice2 = val_dice2

            logger.info('\nEvaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice1, max_dice1))
            logger.info('resnet Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice2, max_dice2))

        """Training"""
        net1.train()
        net2.train()
        logger.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b  = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            img_mask, loss_mask = generate_mask(img_a, patch_size)

            img = img_a * img_mask + img_b * (1 - img_mask)
            lab = lab_a * img_mask + lab_b * (1 - img_mask)

            out1 = net1(img)[0]
            ce_loss1 = F.cross_entropy(out1, lab)
            dice_loss1 = DICE(out1, lab)
            loss1 = (ce_loss1 + dice_loss1) / 2

            out2 = net2(img)[0]
            ce_loss2 = F.cross_entropy(out2, lab)
            dice_loss2 = DICE(out2, lab)
            loss2 = (ce_loss2 + dice_loss2) / 2

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            logger.info("cur epoch: %d step: %d" % (epoch, step+1))
            logger.info("vnet")
            measures.update(out1, lab, ce_loss1, dice_loss1, loss1)
            logger.info("resnet")
            measures.update(out2, lab, ce_loss2, dice_loss2, loss2)
            measures.log(epoch, epoch * len(lab_loader_a) + step)


    return max_dice1

def ema_cutmix(net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader, dae_model_path=None):

    def get_XOR_region(mixout1, mixout2):
        s1 = torch.softmax(mixout1, dim = 1)
        l1 = torch.argmax(s1, dim = 1)

        s2 = torch.softmax(mixout2, dim = 1)
        l2 = torch.argmax(s2, dim = 1)

        diff_mask = (l1 != l2)
        return diff_mask

    """Create Path"""
    save_path = Path(result_dir) / self_train_name
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("EMA_training, save_path: {}".format(str(save_path)))
    measures = CutmixFTMeasures(writer, logger)

    dae_model = None
    if args.use_dae:
        dae_model = create_DAE(lightweight=args.dae_lightweight, trainable=False)
        dae_model.eval()
        dae_path = dae_model_path if dae_model_path else args.dae_model_path
        if dae_path and Path(dae_path).exists():
            dae_state = torch.load(str(dae_path))
            if isinstance(dae_state, dict) and 'net' in dae_state:
                dae_state = dae_state['net']
            dae_model.load_state_dict(dae_state)
            logger.info("Loaded pre-trained DAE model from {}".format(str(dae_path)))
        else:
            logger.warning("DAE model not pre-trained; using random initialization.")

    """Load Model"""
    pretrained_path = Path(result_dir) / 'pretrain'
    load_net_opt(net1, optimizer1, pretrained_path / f'best_ema{label_percent}_pre_vnet.pth')
    load_net_opt(net2, optimizer2, pretrained_path / f'best_ema{label_percent}_pre_resnet.pth')
    load_net_opt(ema_net1, optimizer1, pretrained_path / f'best_ema{label_percent}_pre_vnet.pth')
    logger.info('Loaded from {}'.format(pretrained_path))

    max_dice1 = 0
    max_list1 = None
    max_dice2 = 0
    max_dice3 = 0
    for epoch in tqdm_load(range(1, self_training_epochs+1)):
        measures.reset()
        logger.info('')

        """Testing"""
        if (epoch % 20 == 0) | ((epoch >= 160) & (epoch % 5 ==0)):

            net1.eval()
            net2.eval()

            avg_metric1, _ = test_calculate_metric(net1, test_loader.dataset, s_xy=16, s_z=4)
            avg_metric2, _ = test_calculate_metric(net2, test_loader.dataset, s_xy=16, s_z=4)
            avg_metric3, _ = test_calculate_metric_mean(net1, net2, test_loader.dataset, s_xy=16, s_z=4)

            logger.info('average metric is : {}'.format(avg_metric1))
            logger.info('average metric is : {}'.format(avg_metric2))
            logger.info('mean average metric is : {}'.format(avg_metric3))

            val_dice1 = avg_metric1[0]
            val_dice2 = avg_metric2[0]
            val_dice3 = avg_metric3[0]

            if val_dice1 > max_dice1:
                save_net(net1, str(save_path / f'best_ema_{label_percent}_self.pth'))
                max_dice1 = val_dice1
                max_list1 = avg_metric1

            if val_dice2 > max_dice2:
                save_net(net2, str(save_path / f'best_ema_{label_percent}_self_resnet.pth'))
                max_dice2 = val_dice2


            if val_dice3 > max_dice3:
                save_net(net1, str(save_path / f'best_ema_{label_percent}_self_v.pth'))
                save_net(net2, str(save_path / f'best_ema_{label_percent}_self_r.pth'))

                max_dice3 = val_dice3

            logger.info('\nEvaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice1, max_dice1))
            logger.info('resnet Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice2, max_dice2))
            logger.info('mean Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice3, max_dice3))

        """Training"""
        net1.train()
        net2.train()
        ema_net1.train()
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda([img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])
            """Generate Pseudo Label"""
            with torch.no_grad():
                unimg_a_out_1 = ema_net1(unimg_a)[0]
                unimg_b_out_1 = ema_net1(unimg_b)[0]

                uimg_a_plab = get_cut_mask(unimg_a_out_1, nms=True, connect_mode=connect_mode)
                uimg_b_plab = get_cut_mask(unimg_b_out_1, nms=True, connect_mode=connect_mode)


                img_mask, loss_mask = generate_mask(img_a, patch_size)


            """Mix input"""
            net3_input_l = unimg_a * img_mask + img_b * (1 - img_mask)
            net3_input_unlab = img_a * img_mask + unimg_b * (1 - img_mask)

            """BCP"""
            """Supervised Loss"""
            mix_lab_out = net1(net3_input_l)
            mix_output_l = mix_lab_out[0]
            loss_1 = mix_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True)

            """Unsupervised Loss"""
            mix_unlab_out = net1(net3_input_unlab)
            mix_output_2 = mix_unlab_out[0]
            loss_2 = mix_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask)


            """Supervised Loss"""
            mix_output_l_2 = net2(net3_input_l)[0]
            loss_1_2 = mix_loss(mix_output_l_2, uimg_a_plab.long(), lab_b, loss_mask, unlab=True)

            """Unsupervised Loss"""
            mix_output_2_2 = net2(net3_input_unlab)[0]
            loss_2_2 = mix_loss(mix_output_2_2, lab_a, uimg_b_plab.long(), loss_mask)

            """SDCL"""

            with torch.no_grad():
                diff_mask1 = get_XOR_region(mix_output_l, mix_output_l_2)
                diff_mask2 = get_XOR_region(mix_output_2, mix_output_2_2)

            net1_mse_loss_lab = mix_mse_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)
            net1_kl_loss_lab = mix_max_kl_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)

            net1_mse_loss_unlab = mix_mse_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask, diff_mask=diff_mask2)
            net1_kl_loss_unlab = mix_max_kl_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask, diff_mask=diff_mask2)

            net2_mse_loss_lab = mix_mse_loss(mix_output_l_2, uimg_a_plab.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)
            net2_kl_loss_lab = mix_max_kl_loss(mix_output_l_2, uimg_a_plab.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)

            net2_mse_loss_unlab = mix_mse_loss(mix_output_2_2, lab_a, uimg_b_plab.long(), loss_mask, diff_mask=diff_mask2)
            net2_kl_loss_unlab = mix_max_kl_loss(mix_output_2_2, lab_a, uimg_b_plab.long(), loss_mask, diff_mask=diff_mask2)

            sam_loss_1 = torch.tensor(0.0).cuda()
            sam_loss_2 = torch.tensor(0.0).cuda()
            if args.use_sam:
                try:
                    mix_output_l_soft = F.softmax(mix_output_l, dim=1)
                    mix_output_2_soft = F.softmax(mix_output_2, dim=1)
                    mix_output_l_2_soft = F.softmax(mix_output_l_2, dim=1)
                    mix_output_2_2_soft = F.softmax(mix_output_2_2, dim=1)

                    sam_mask_l, unc_l = semisam_branch(
                        net3_input_l,
                        mix_output_l_soft[:, 1:2, :, :, :],
                        generalist='SAM-Med3D',
                        prompt=args.sam_prompt,
                    )
                    sam_mask_u, unc_u = semisam_branch(
                        net3_input_unlab,
                        mix_output_2_soft[:, 1:2, :, :, :],
                        generalist='SAM-Med3D',
                        prompt=args.sam_prompt,
                    )
                    sam_branch = get_sam_branch()
                    if not getattr(sam_branch, '_initialized', False):
                        raise RuntimeError(
                            'SAM-Med3D is not initialized. Check the checkpoint path and segment_anything dependencies.'
                        )
                    sam_target_l = torch.cat((1 - sam_mask_l, sam_mask_l), dim=1)
                    sam_target_u = torch.cat((1 - sam_mask_u, sam_mask_u), dim=1)
                    sam_weight = get_sam_consistency_weight(epoch, self_training_epochs)

                    if args.sam_prompt == 'unc':
                        sam_loss_l = torch.mean(((mix_output_l_soft - sam_target_l) ** 2) * (1 - unc_l)) / (torch.mean(1 - unc_l) + 1e-8)
                        sam_loss_u = torch.mean(((mix_output_2_soft - sam_target_u) ** 2) * (1 - unc_u)) / (torch.mean(1 - unc_u) + 1e-8)
                        sam_loss_l_2 = torch.mean(((mix_output_l_2_soft - sam_target_l) ** 2) * (1 - unc_l)) / (torch.mean(1 - unc_l) + 1e-8)
                        sam_loss_u_2 = torch.mean(((mix_output_2_2_soft - sam_target_u) ** 2) * (1 - unc_u)) / (torch.mean(1 - unc_u) + 1e-8)
                        unc_reg = 0.1 * (torch.mean(unc_l) + torch.mean(unc_u))
                        sam_loss_1 = sam_weight * (sam_loss_l + sam_loss_u + unc_reg)
                        sam_loss_2 = sam_weight * (sam_loss_l_2 + sam_loss_u_2 + unc_reg)
                    else:
                        sam_loss_1 = sam_weight * (
                            torch.mean((mix_output_l_soft - sam_target_l) ** 2)
                            + torch.mean((mix_output_2_soft - sam_target_u) ** 2)
                        )
                        sam_loss_2 = sam_weight * (
                            torch.mean((mix_output_l_2_soft - sam_target_l) ** 2)
                            + torch.mean((mix_output_2_2_soft - sam_target_u) ** 2)
                        )
                except Exception as e:
                    if not args.sam_skip_on_error:
                        raise RuntimeError(
                            'SAM branch failed. Fix SAM-Med3D availability or rerun with '
                            '--sam_skip_on_error 1 to train without SAM loss.'
                        ) from e
                    logger.warning("SAM branch error: {}, skipping SAM loss because --sam_skip_on_error=1".format(e))
                    sam_loss_1 = torch.tensor(0.0).cuda()
                    sam_loss_2 = torch.tensor(0.0).cuda()

            dae_loss_1 = torch.tensor(0.0).cuda()
            dae_loss_2 = torch.tensor(0.0).cuda()
            if args.use_dae and dae_model is not None:
                try:
                    with torch.no_grad():
                        ema_output_a_soft = F.softmax(unimg_a_out_1, dim=1)
                        ema_output_b_soft = F.softmax(unimg_b_out_1, dim=1)
                        dae_input_a = ema_output_a_soft[:, 1:2, :, :, :]
                        dae_input_b = ema_output_b_soft[:, 1:2, :, :, :]

                        dae_output_a, _ = dae_model(dae_input_a)
                        dae_output_b, _ = dae_model(dae_input_b)
                        dae_preds_a = torch.sigmoid(dae_output_a)
                        dae_preds_b = torch.sigmoid(dae_output_b)

                        certainty_a = torch.exp(-args.dae_gamma * ((dae_preds_a - dae_input_a) ** 2))
                        certainty_b = torch.exp(-args.dae_gamma * ((dae_preds_b - dae_input_b) ** 2))
                        certainty_a = certainty_a.expand(-1, 2, -1, -1, -1)
                        certainty_b = certainty_b.expand(-1, 2, -1, -1, -1)
                        dae_target_a = torch.cat([1 - dae_preds_a, dae_preds_a], dim=1)
                        dae_target_b = torch.cat([1 - dae_preds_b, dae_preds_b], dim=1)

                    mix_output_l_soft = F.softmax(mix_output_l, dim=1)
                    mix_output_2_soft = F.softmax(mix_output_2, dim=1)
                    mix_output_l_2_soft = F.softmax(mix_output_l_2, dim=1)
                    mix_output_2_2_soft = F.softmax(mix_output_2_2, dim=1)
                    mask = loss_mask.unsqueeze(1).float()
                    patch_mask = (1 - loss_mask).unsqueeze(1).float()
                    dae_weight = get_dae_consistency_weight(epoch)

                    dae_dist_l = (mix_output_l_soft - dae_target_a.detach()) ** 2
                    dae_dist_u = (mix_output_2_soft - dae_target_b.detach()) ** 2
                    dae_loss_l = torch.sum(certainty_a * dae_dist_l * mask) / (torch.sum(certainty_a * mask) + 1e-8)
                    dae_loss_u = torch.sum(certainty_b * dae_dist_u * patch_mask) / (torch.sum(certainty_b * patch_mask) + 1e-8)
                    dae_loss_1 = dae_weight * (dae_loss_l + dae_loss_u)

                    dae_dist_l_2 = (mix_output_l_2_soft - dae_target_a.detach()) ** 2
                    dae_dist_u_2 = (mix_output_2_2_soft - dae_target_b.detach()) ** 2
                    dae_loss_l_2 = torch.sum(certainty_a * dae_dist_l_2 * mask) / (torch.sum(certainty_a * mask) + 1e-8)
                    dae_loss_u_2 = torch.sum(certainty_b * dae_dist_u_2 * patch_mask) / (torch.sum(certainty_b * patch_mask) + 1e-8)
                    dae_loss_2 = dae_weight * (dae_loss_l_2 + dae_loss_u_2)
                except Exception as e:
                    logger.warning("DAE branch error: {}, skipping DAE loss".format(e))
                    dae_loss_1 = torch.tensor(0.0).cuda()
                    dae_loss_2 = torch.tensor(0.0).cuda()

            loss1 = loss_1 + loss_2 + 0.3 * (net1_mse_loss_lab + net1_mse_loss_unlab) + 0.1 * (net1_kl_loss_lab + net1_kl_loss_unlab) + sam_loss_1 + dae_loss_1

            loss2 = loss_1_2 + loss_2_2 + 0.3 * (net2_mse_loss_lab + net2_mse_loss_unlab) + 0.1 * (net2_kl_loss_lab + net2_kl_loss_unlab) + sam_loss_2 + dae_loss_2

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            update_ema_variables(net1, ema_net1, alpha)

            logger.info("loss_1: %.4f, loss_2: %.4f, net1_mse_loss_lab: %.4f, net1_mse_loss_unlab: %.4f, net1_kl_loss_lab: %.4f, net1_kl_loss_unlab: %.4f," % 
                (loss_1.item(), loss_2.item(), net1_mse_loss_lab.item(), net1_mse_loss_unlab.item(),
                    net1_kl_loss_lab.item(), net1_kl_loss_unlab.item()))
            logger.info("sam_loss: %.4f, dae_loss: %.4f" % (sam_loss_1.item(), dae_loss_1.item()))

        if epoch == self_training_epochs:
            save_net(net1, str(save_path / f'best_ema_{label_percent}_self_latest.pth'))
    return max_dice1, max_list1

def test_model(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    load_path = Path(result_dir) / self_train_name
    load_net(net1, load_path / f'best_ema_{label_percent}_self.pth')
    load_net(net2, load_path / f'best_ema_{label_percent}_self_resnet.pth')
    print('Successful Loaded')
    avg_metric, _ = test_calculate_metric(net1, test_loader.dataset, s_xy=16, s_z=4)
    avg_metric2, _ = test_calculate_metric(net2, test_loader.dataset, s_xy=16, s_z=4)
    avg_metric3, _ = test_calculate_metric_mean(net1, net2, test_loader.dataset, s_xy=16, s_z=4)
    print(avg_metric)
    print(avg_metric2)
    print(avg_metric3)


if __name__ == '__main__':
    try:
        net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader = get_ema_model_and_dataloader(data_root, split_name, batch_size, lr, labelp=label_percent)
        pretrained_path = Path(result_dir) / 'pretrain'
        if args.skip_pretrain:
            required_files = [
                pretrained_path / f'best_ema{label_percent}_pre_vnet.pth',
                pretrained_path / f'best_ema{label_percent}_pre_resnet.pth',
            ]
            missing_files = [str(path) for path in required_files if not path.exists()]
            if missing_files:
                raise FileNotFoundError("Skipping pre-training requires existing weights: {}".format(missing_files))
            print("Skip pre-training and use weights from {}".format(str(pretrained_path)))
        else:
            pretrain(net1, net2, optimizer1, optimizer2, lab_loader_a, lab_loader_b, test_loader)

        dae_model_path = None
        if args.use_dae:
            dae_save_path = Path(result_dir) / 'dae_pretrain'
            dae_save_path.mkdir(exist_ok=True)
            if args.skip_dae_pretrain:
                default_dae_path = dae_save_path / 'best_dae_model.pth'
                if args.dae_model_path and Path(args.dae_model_path).exists():
                    dae_model_path = Path(args.dae_model_path)
                    print("Use DAE model from {}".format(str(dae_model_path)))
                elif default_dae_path.exists():
                    dae_model_path = default_dae_path
                    print("Skip DAE pre-training and use {}".format(str(dae_model_path)))
                else:
                    print("DAE model not found; self-training will use random DAE initialization.")
            else:
                logger, _ = config_log(dae_save_path, tensorboard=False)
                logger.info("Stage: DAE pre-training on labeled pancreas data")
                dae_model_path = pretrain_dae(lab_loader_a, dae_save_path)

        seed_reproducer(seed = seed_test)
        ema_cutmix(net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader, dae_model_path=dae_model_path)
        test_model(net1, net2, test_loader)

    except Exception as e:
        if logger is not None:
            logger.exception("BUG FOUNDED ! ! !")
        else:
            raise
