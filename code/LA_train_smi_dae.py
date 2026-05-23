import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.LADataset import LAHeart
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables
from utils.LA_utils import to_cuda
from utils.BCP_utils import *
from pancreas.losses import *

from pancreas.Vnet import VNet
from networks.ResVNet import ResVNet

# SAM-Med3D integration for semi-supervised learning enhancement
from semisam_branch import semisam_branch, compute_sam_consistency_loss, compute_sam_dice_loss, get_sam_branch

# DAE (Denoising Auto-Encoder) for pseudo-label refinement and uncertainty estimation
from networks.VNetDAE import VNetDAE, LightweightDAE, compute_dae_uncertainty

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='code/Datasets/la/data_split', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='SDCL', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=1e-3, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1345, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default=10.0, help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
# -- setting of SAM-enhanced semi-supervised learning
parser.add_argument('--use_sam', type=int, default=1, help='whether to use SAM-Med3D enhancement')
parser.add_argument('--sam_prompt', type=str, default='unc', help='SAM prompt strategy: unc, centroid, random')
parser.add_argument('--sam_weight', type=float, default=0.1, help='weight for SAM consistency loss')
parser.add_argument('--sam_rampup', type=float, default=40.0, help='rampup for SAM loss weight')
parser.add_argument('--sam_skip_on_error', type=int, default=0, help='skip SAM loss when SAM-Med3D is unavailable or errors')
# -- setting for skipping pre-training
parser.add_argument('--skip_pretrain', type=int, default=0, help='whether to skip pre-training (use existing weights)')
parser.add_argument('--pretrain_path', type=str, default='', help='path to pre-trained weights (if skip_pretrain=1)')
# -- setting for DAE (Denoising Auto-Encoder) enhanced semi-supervised learning
parser.add_argument('--use_dae', type=int, default=1, help='whether to use DAE for pseudo-label denoising')
parser.add_argument('--dae_model_path', type=str, default='', help='path to pre-trained DAE model (optional)')
parser.add_argument('--dae_gamma', type=float, default=1.0, help='uncertainty weight factor for DAE')
parser.add_argument('--dae_weight', type=float, default=0.5, help='weight for DAE consistency loss')
parser.add_argument('--dae_rampup', type=float, default=40.0, help='rampup epochs for DAE loss weight')
parser.add_argument('--dae_lightweight', type=int, default=0, help='use lightweight DAE model')
parser.add_argument('--dae_pretrain_epochs', type=int, default=100, help='epochs for DAE pre-training')
parser.add_argument('--dae_pretrain_lr', type=float, default=1e-3, help='learning rate for DAE pre-training')
parser.add_argument('--dae_noise_ratio', type=float, default=0.2, help='noise ratio for DAE training')
parser.add_argument('--skip_dae_pretrain', type=int, default=0, help='whether to skip DAE pre-training')
args = parser.parse_args()


def create_Vnet(ema=False):
    net = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def create_ResVnet(ema=False):
    net = ResVNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def create_DAE(lightweight=False, trainable=False):
    """创建DAE模型用于伪标签去噪
    
    Args:
        lightweight: 是否使用轻量级DAE
        trainable: 是否可训练（预训练时为True，推理时为False）
    """
    if lightweight:
        # 轻量级DAE：输入1通道，输出1通道（前景概率）
        net = LightweightDAE(n_channels=1, n_classes=1, n_filters=8, normalization='instancenorm')
    else:
        # 标准DAE：输入1通道，输出1通道（前景概率）
        net = VNetDAE(n_channels=1, n_classes=1, n_filters=16, normalization='instancenorm', 
                      has_dropout=False, input_size=patch_size, is_LS_noise=True, emb_dim=256)
    net = nn.DataParallel(net)
    model = net.cuda()
    if not trainable:
        for param in model.parameters():
            param.detach_()  # DAE参数不参与主网络的梯度更新
    return model


def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def get_cut_mask_two(out1, out2, thres=0.5, nms=0):
    probs1 = F.softmax(out1, 1)
    probs2 = F.softmax(out2, 1)
    probs = (probs1 + probs2) / 2

    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_sam_consistency_weight(epoch, max_epoch):
    """
    SAM consistency weight schedule:
    - Starts high at beginning (when model is uncertain, SAM provides strong guidance)
    - Decreases during the first sam_rampup epochs as the model becomes more confident
    """
    if args.sam_rampup <= 0:
        return args.sam_weight
    current = np.clip(epoch - 1, 0.0, args.sam_rampup)
    return args.sam_weight * ramps.cosine_rampdown(current, args.sam_rampup)


def get_dae_consistency_weight(epoch):
    """
    DAE consistency weight schedule:
    使用rampup策略，随着训练进行逐渐增加DAE的权重
    """
    return args.dae_weight * ramps.sigmoid_rampup(epoch, args.dae_rampup)


train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def get_XOR_region(mixout1, mixout2):
    s1 = torch.softmax(mixout1, dim=1)
    l1 = torch.argmax(s1, dim=1)

    s2 = torch.softmax(mixout2, dim=1)
    l2 = torch.argmax(s2, dim=1)

    diff_mask = (l1 != l2)
    return diff_mask


def cmp_dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def add_noise_to_label(label, noise_ratio=0.2):
    """给标签添加噪声用于DAE训练
    
    Args:
        label: 输入标签 (B, 1, D, H, W)，值为0或1
        noise_ratio: 噪声比例
    
    Returns:
        noisy_label: 加噪后的标签
    """
    noise = torch.rand_like(label.float())
    # 随机翻转一些像素
    flip_mask = noise < noise_ratio
    noisy_label = label.float().clone()
    noisy_label[flip_mask] = 1.0 - noisy_label[flip_mask]
    
    # 添加高斯噪声使其更平滑
    gaussian_noise = torch.randn_like(noisy_label) * 0.1
    noisy_label = torch.clamp(noisy_label + gaussian_noise, 0.0, 1.0)
    
    return noisy_label


def pre_train_dae(args, snapshot_path):
    """DAE预训练：在有标签数据上训练DAE学习解剖先验
    
    训练目标：输入加噪的标签，重建干净的标签
    这样DAE可以学习到正常的解剖结构分布
    """
    logging.info("="*50)
    logging.info("Starting DAE Pre-training on labeled data...")
    logging.info("="*50)
    
    # 创建可训练的DAE模型
    dae_model = create_DAE(lightweight=args.dae_lightweight, trainable=True)
    
    # 数据加载
    c_batch_size = 2
    trainset_lab = LAHeart(train_data_path, "code/Datasets/la/data_split", split='train_lab', logging=logging)
    lab_loader = DataLoader(trainset_lab, batch_size=c_batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    # 优化器
    optimizer_dae = optim.Adam(dae_model.parameters(), lr=args.dae_pretrain_lr)
    scheduler_dae = optim.lr_scheduler.CosineAnnealingLR(optimizer_dae, args.dae_pretrain_epochs)
    
    # 损失函数
    bce_loss = nn.BCEWithLogitsLoss()
    
    dae_model.train()
    best_loss = float('inf')
    
    logging.info(f"DAE pre-training for {args.dae_pretrain_epochs} epochs")
    logging.info(f"Noise ratio: {args.dae_noise_ratio}")
    
    iterator = tqdm(range(1, args.dae_pretrain_epochs + 1), ncols=70, desc="DAE Pre-train")
    
    for epoch in iterator:
        epoch_loss = 0.0
        epoch_dice = 0.0
        num_batches = 0
        
        for step, (img, lab) in enumerate(lab_loader):
            img, lab = img.cuda(), lab.cuda()
            
            # 将标签转换为单通道 (B, 1, D, H, W)
            # lab原始形状是 (B, D, H, W)，需要unsqueeze
            if lab.dim() == 4:
                lab = lab.unsqueeze(1).float()  # (B, 1, D, H, W)
            else:
                lab = lab.float()
            
            # 添加噪声到标签
            noisy_lab = add_noise_to_label(lab, noise_ratio=args.dae_noise_ratio)
            
            # DAE前向传播：输入加噪标签，输出重建标签
            dae_output, _ = dae_model(noisy_lab)
            
            # 计算损失：BCE + Dice
            loss_bce = bce_loss(dae_output, lab)
            
            # Dice损失
            dae_pred = torch.sigmoid(dae_output)
            loss_dice = cmp_dice_loss(dae_pred, lab)
            
            loss = loss_bce + loss_dice
            
            optimizer_dae.zero_grad()
            loss.backward()
            optimizer_dae.step()
            
            epoch_loss += loss.item()
            epoch_dice += (1 - loss_dice.item())
            num_batches += 1
        
        scheduler_dae.step()
        
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        
        if epoch % 10 == 0:
            logging.info(f'DAE Epoch {epoch}: loss={avg_loss:.4f}, dice={avg_dice:.4f}, lr={scheduler_dae.get_last_lr()[0]:.6f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(snapshot_path, 'best_dae_model.pth')
            torch.save(dae_model.state_dict(), save_path)
            logging.info(f"Save best DAE model to {save_path}, loss={avg_loss:.4f}")
    
    # 保存最终模型
    final_save_path = os.path.join(snapshot_path, 'final_dae_model.pth')
    torch.save(dae_model.state_dict(), final_save_path)
    logging.info(f"DAE pre-training finished. Final model saved to {final_save_path}")
    
    return os.path.join(snapshot_path, 'best_dae_model.pth')


def pre_train(args, snapshot_path):
    model = create_Vnet()
    model2 = create_ResVnet()

    c_batch_size = 2
    trainset_lab_a = LAHeart(train_data_path, "code/Datasets/la/data_split", split='train_lab', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "code/Datasets/la/data_split", split='train_lab', reverse=True, logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)



    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    model2.train()
    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0
    best_dice2 = 0
    max_epoch = 81
    iterator = tqdm(range(1, max_epoch), ncols=70)
    for epoch_num in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs, _ = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            outputs2, _ = model2(volume_batch)
            loss_ce2 = F.cross_entropy(outputs2, label_batch)
            loss_dice2 = DICE(outputs2, label_batch)
            loss2 = (loss_ce2 + loss_dice2) / 2

            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            logging.info(
                'iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num >= pre_max_iterations:
                logging.info("Pre-training reached max iterations: {}".format(pre_max_iterations))
                return

        if epoch_num % 5 == 0:
            model.eval()
            dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=18, stride_z=4, data_root=train_data_path)
            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(snapshot_path, 'best_model.pth'.format(args.model))
                save_net_opt(model, optimizer, save_mode_path, epoch_num)
                save_net_opt(model, optimizer, save_best_path, epoch_num)
                logging.info("save best model to {}".format(save_mode_path))

            model.train()

            model2.eval()
            dice_sample2 = test_3d_patch.var_all_case_LA(model2, num_classes=num_classes, patch_size=patch_size,
                                                         stride_xy=18, stride_z=4, data_root=train_data_path)
            if dice_sample2 > best_dice2:
                best_dice2 = round(dice_sample2, 4)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}_resnet.pth'.format(iter_num, best_dice2))
                save_best_path = os.path.join(snapshot_path, 'best_model_resnet.pth'.format(args.model))
                save_net_opt(model2, optimizer2, save_mode_path, epoch_num)
                save_net_opt(model2, optimizer2, save_best_path, epoch_num)
                logging.info("save best resnet model to {}".format(save_mode_path))
            model2.train()



def self_train(args, pre_snapshot_path, self_snapshot_path, dae_model_path=None):
    model1 = create_Vnet()
    model2 = create_ResVnet()
    ema_model1 = create_Vnet(ema=True).cuda()
    
    # 初始化DAE模型用于伪标签去噪和不确定性估计
    dae_model = None
    if args.use_dae:
        dae_model = create_DAE(lightweight=args.dae_lightweight, trainable=False)
        dae_model.eval()  # DAE始终在eval模式
        logging.info("DAE model initialized for pseudo-label denoising")
        
        # 加载预训练的DAE模型（优先使用传入的路径，否则使用命令行参数）
        dae_path = dae_model_path if dae_model_path else args.dae_model_path
        if dae_path and os.path.exists(dae_path):
            dae_model.load_state_dict(torch.load(dae_path))
            logging.info(f"Loaded pre-trained DAE model from {dae_path}")
        else:
            logging.warning("DAE model not pre-trained! Using random initialization. "
                          "This may lead to poor uncertainty estimation. "
                          "Consider running with --skip_dae_pretrain=0")


    c_batch_size = 2
    trainset_lab_a = LAHeart(train_data_path, "code/Datasets/la/data_split", split='train_lab', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "code/Datasets/la/data_split", split='train_lab', reverse=True, logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_a = LAHeart(train_data_path, "code/Datasets/la/data_split", split='train_unlab', logging=logging)
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_b = LAHeart(train_data_path, "code/Datasets/la/data_split", split='train_unlab', reverse=True, logging=logging)
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)



    optimizer = optim.Adam(model1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)


    pretrained_model = os.path.join(pre_snapshot_path, 'best_model.pth')
    pretrained_model2 = os.path.join(pre_snapshot_path, 'best_model_resnet.pth')

    load_net_opt(model1, optimizer, pretrained_model)
    load_net_opt(model2, optimizer2, pretrained_model2)

    load_net_opt(ema_model1, optimizer, pretrained_model)


    model1.train()
    model2.train()
    ema_model1.train()

    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0
    best_dice2 = 0
    mean_best_dice = 0
    max_epoch = 276
    iterator = tqdm(range(1, max_epoch), ncols=70)
    for epoch in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(
                zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda(
                [img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])

            with torch.no_grad():

                unoutput_a_1, _ = ema_model1(unimg_a)
                unoutput_b_1, _ = ema_model1(unimg_b)


                plab_a = get_cut_mask(unoutput_a_1, nms=1)
                plab_b = get_cut_mask(unoutput_b_1, nms=1)

                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            mixl_img = unimg_a * img_mask + img_b * (1 - img_mask)
            mixu_img = img_a * img_mask + unimg_b * (1 - img_mask)


            outputs_l, _ = model1(mixl_img)
            outputs_u, _ = model1(mixu_img)
            loss_l = mix_loss(outputs_l, plab_a.long(), lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            loss_u = mix_loss(outputs_u, lab_a, plab_b.long(), loss_mask, u_weight=args.u_weight)

            outputs_l_2, _ = model2(mixl_img)
            outputs_u_2, _ = model2(mixu_img)
            loss_l_2 = mix_loss(outputs_l_2, plab_a.long(), lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            loss_u_2 = mix_loss(outputs_u_2, lab_a, plab_b.long(), loss_mask, u_weight=args.u_weight)

            with torch.no_grad():
                diff_mask1 = get_XOR_region(outputs_l, outputs_l_2)
                diff_mask2 = get_XOR_region(outputs_u, outputs_u_2)

            net1_mse_loss_lab = mix_mse_loss(outputs_l, plab_a.long(), lab_b, loss_mask, unlab=True,
                                             diff_mask=diff_mask1)
            net1_kl_loss_lab = mix_max_kl_loss(outputs_l, plab_a.long(), lab_b, loss_mask, unlab=True,
                                               diff_mask=diff_mask1)

            net1_mse_loss_unlab = mix_mse_loss(outputs_u, lab_a, plab_b.long(), loss_mask, diff_mask=diff_mask2)
            net1_kl_loss_unlab = mix_max_kl_loss(outputs_u, lab_a, plab_b.long(), loss_mask, diff_mask=diff_mask2)

            net2_mse_loss_lab = mix_mse_loss(outputs_l_2, plab_a.long(), lab_b, loss_mask, unlab=True,
                                             diff_mask=diff_mask1)
            net2_kl_loss_lab = mix_max_kl_loss(outputs_l_2, plab_a.long(), lab_b, loss_mask, unlab=True,
                                               diff_mask=diff_mask1)

            net2_mse_loss_unlab = mix_mse_loss(outputs_u_2, lab_a, plab_b.long(), loss_mask, diff_mask=diff_mask2)
            net2_kl_loss_unlab = mix_max_kl_loss(outputs_u_2, lab_a, plab_b.long(), loss_mask, diff_mask=diff_mask2)

            # ========== SAM-Med3D Enhanced Semi-Supervised Learning ==========
            sam_loss_1 = torch.tensor(0.0).cuda()
            sam_loss_2 = torch.tensor(0.0).cuda()
            
            if args.use_sam:
                try:
                    # Get softmax predictions for unlabeled data
                    outputs_l_soft = F.softmax(outputs_l, dim=1)
                    outputs_u_soft = F.softmax(outputs_u, dim=1)
                    outputs_l_2_soft = F.softmax(outputs_l_2, dim=1)
                    outputs_u_2_soft = F.softmax(outputs_u_2, dim=1)
                    
                    # Apply SAM-Med3D to get refined segmentation and uncertainty
                    # For mixl_img (unlabeled + labeled mix)
                    sam_mask_l, unc_l = semisam_branch(
                        mixl_img, outputs_l_soft[:, 1:2, :, :, :], 
                        generalist='SAM-Med3D', prompt=args.sam_prompt
                    )
                    
                    # For mixu_img (labeled + unlabeled mix)  
                    sam_mask_u, unc_u = semisam_branch(
                        mixu_img, outputs_u_soft[:, 1:2, :, :, :],
                        generalist='SAM-Med3D', prompt=args.sam_prompt
                    )

                    sam_branch = get_sam_branch()
                    if not getattr(sam_branch, '_initialized', False):
                        raise RuntimeError(
                            'SAM-Med3D is not initialized. Check the checkpoint path and segment_anything dependencies.'
                        )
                    
                    # Compute SAM consistency loss for model1
                    sam_samseg_l = torch.cat((1 - sam_mask_l, sam_mask_l), dim=1)
                    sam_samseg_u = torch.cat((1 - sam_mask_u, sam_mask_u), dim=1)
                    
                    # SAM consistency weight (high at start, decreasing)
                    sam_weight = get_sam_consistency_weight(epoch, max_epoch)
                    
                    if args.sam_prompt == 'unc':
                        # Uncertainty-weighted SAM consistency loss
                        sam_dist_l = (outputs_l_soft - sam_samseg_l) ** 2
                        sam_dist_u = (outputs_u_soft - sam_samseg_u) ** 2
                        
                        # Weight by inverse uncertainty (focus on confident predictions)
                        sam_loss_l_item = torch.mean(sam_dist_l * (1 - unc_l)) / (torch.mean(1 - unc_l) + 1e-8)
                        sam_loss_u_item = torch.mean(sam_dist_u * (1 - unc_u)) / (torch.mean(1 - unc_u) + 1e-8)
                        
                        # Add uncertainty regularization
                        sam_loss_1 = sam_weight * (sam_loss_l_item + sam_loss_u_item + 0.1 * (torch.mean(unc_l) + torch.mean(unc_u)))
                    else:
                        # Standard MSE consistency
                        sam_loss_1 = sam_weight * (
                            torch.mean((outputs_l_soft - sam_samseg_l) ** 2) +
                            torch.mean((outputs_u_soft - sam_samseg_u) ** 2)
                        )
                    
                    # SAM consistency loss for model2
                    if args.sam_prompt == 'unc':
                        sam_dist_l_2 = (outputs_l_2_soft - sam_samseg_l) ** 2
                        sam_dist_u_2 = (outputs_u_2_soft - sam_samseg_u) ** 2
                        
                        sam_loss_l_2_item = torch.mean(sam_dist_l_2 * (1 - unc_l)) / (torch.mean(1 - unc_l) + 1e-8)
                        sam_loss_u_2_item = torch.mean(sam_dist_u_2 * (1 - unc_u)) / (torch.mean(1 - unc_u) + 1e-8)
                        
                        sam_loss_2 = sam_weight * (sam_loss_l_2_item + sam_loss_u_2_item + 0.1 * (torch.mean(unc_l) + torch.mean(unc_u)))
                    else:
                        sam_loss_2 = sam_weight * (
                            torch.mean((outputs_l_2_soft - sam_samseg_l) ** 2) +
                            torch.mean((outputs_u_2_soft - sam_samseg_u) ** 2)
                        )
                        
                except Exception as e:
                    if not args.sam_skip_on_error:
                        raise RuntimeError(
                            'SAM branch failed. Fix SAM-Med3D availability or rerun with '
                            '--sam_skip_on_error 1 to train without SAM loss.'
                        ) from e
                    logging.warning(f"SAM branch error: {e}, skipping SAM loss because --sam_skip_on_error=1")
                    sam_loss_1 = torch.tensor(0.0).cuda()
                    sam_loss_2 = torch.tensor(0.0).cuda()
            # ========== End SAM Enhancement ==========

            # ========== DAE Enhanced Semi-Supervised Learning ==========
            dae_loss_1 = torch.tensor(0.0).cuda()
            dae_loss_2 = torch.tensor(0.0).cuda()
            dae_certainty_mask = None
            
            if args.use_dae and dae_model is not None:
                try:
                    with torch.no_grad():
                        # 获取EMA模型对无标签数据的softmax预测
                        ema_output_a_soft = F.softmax(unoutput_a_1, dim=1)
                        ema_output_b_soft = F.softmax(unoutput_b_1, dim=1)
                        
                        # 将预测转换为单通道（argmax后的标签）作为DAE输入
                        # 使用前景类的概率作为输入
                        dae_input_a = ema_output_a_soft[:, 1:2, :, :, :]  # (B, 1, D, H, W)
                        dae_input_b = ema_output_b_soft[:, 1:2, :, :, :]
                        
                        # DAE去噪 - 输出是1通道
                        dae_output_a, _ = dae_model(dae_input_a)
                        dae_output_b, _ = dae_model(dae_input_b)
                        
                        # 计算不确定性：DAE输出与原始预测的L2距离
                        # dae_output是logits，需要sigmoid转为概率
                        dae_preds_a = torch.sigmoid(dae_output_a)  # (B, 1, D, H, W)
                        dae_preds_b = torch.sigmoid(dae_output_b)  # (B, 1, D, H, W)
                        
                        # 不确定性：DAE预测与原始预测的差异
                        uncertainty_a = (dae_preds_a - dae_input_a) ** 2  # (B, 1, D, H, W)
                        uncertainty_b = (dae_preds_b - dae_input_b) ** 2
                        
                        # 确定性掩码：低不确定性区域权重高
                        certainty_a = torch.exp(-args.dae_gamma * uncertainty_a)  # (B, 1, D, H, W)
                        certainty_b = torch.exp(-args.dae_gamma * uncertainty_b)
                        
                        # 扩展到2通道以匹配outputs (num_classes=2)
                        certainty_a_expanded = certainty_a.expand(-1, num_classes, -1, -1, -1)  # (B, 2, D, H, W)
                        certainty_b_expanded = certainty_b.expand(-1, num_classes, -1, -1, -1)
                        
                        # 构建DAE去噪后的目标（2通道: 背景+前景）
                        dae_target_a = torch.cat([1 - dae_preds_a, dae_preds_a], dim=1)  # (B, 2, D, H, W)
                        dae_target_b = torch.cat([1 - dae_preds_b, dae_preds_b], dim=1)
                    
                    # 获取当前输出的softmax
                    outputs_l_soft = F.softmax(outputs_l, dim=1)  # (B, 2, D, H, W)
                    outputs_u_soft = F.softmax(outputs_u, dim=1)
                    outputs_l_2_soft = F.softmax(outputs_l_2, dim=1)
                    outputs_u_2_soft = F.softmax(outputs_u_2, dim=1)
                    
                    # DAE一致性权重
                    dae_weight = get_dae_consistency_weight(epoch)
                    
                    # 计算DAE加权一致性损失
                    # Model1的DAE一致性损失
                    dae_dist_l = (outputs_l_soft - dae_target_a.detach()) ** 2
                    dae_dist_u = (outputs_u_soft - dae_target_b.detach()) ** 2
                    
                    # 使用确定性掩码加权：只在高确定性区域强制一致性
                    dae_loss_l = torch.sum(certainty_a_expanded * dae_dist_l * loss_mask) / (torch.sum(certainty_a_expanded * loss_mask) + 1e-8)
                    dae_loss_u = torch.sum(certainty_b_expanded * dae_dist_u * (1 - loss_mask)) / (torch.sum(certainty_b_expanded * (1 - loss_mask)) + 1e-8)
                    
                    dae_loss_1 = dae_weight * (dae_loss_l + dae_loss_u)
                    
                    # Model2的DAE一致性损失
                    dae_dist_l_2 = (outputs_l_2_soft - dae_target_a.detach()) ** 2
                    dae_dist_u_2 = (outputs_u_2_soft - dae_target_b.detach()) ** 2
                    
                    dae_loss_l_2 = torch.sum(certainty_a_expanded * dae_dist_l_2 * loss_mask) / (torch.sum(certainty_a_expanded * loss_mask) + 1e-8)
                    dae_loss_u_2 = torch.sum(certainty_b_expanded * dae_dist_u_2 * (1 - loss_mask)) / (torch.sum(certainty_b_expanded * (1 - loss_mask)) + 1e-8)
                    
                    dae_loss_2 = dae_weight * (dae_loss_l_2 + dae_loss_u_2)
                    
                    # 保存确定性掩码用于可能的可视化
                    dae_certainty_mask = (certainty_a, certainty_b)
                    
                except Exception as e:
                    logging.warning(f"DAE branch error: {e}, skipping DAE loss")
                    dae_loss_1 = torch.tensor(0.0).cuda()
                    dae_loss_2 = torch.tensor(0.0).cuda()
            # ========== End DAE Enhancement ==========

            loss = (loss_l + loss_u) + 0.5 * (net1_mse_loss_lab + net1_mse_loss_unlab) + 0.05 * (
                        net1_kl_loss_lab + net1_kl_loss_unlab) + sam_loss_1 + dae_loss_1

            loss_2 = (loss_l_2 + loss_u_2) + 0.5 * (net2_mse_loss_lab + net2_mse_loss_unlab) + 0.05 * (
                        net2_kl_loss_lab + net2_kl_loss_unlab) + sam_loss_2 + dae_loss_2


            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()

            logging.info('epoch %d iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f \
               net1_mse_loss_lab: %.4f, net1_mse_loss_unlab: %.4f, net1_kl_loss_lab: %.4f, net1_kl_loss_unlab: %.4f, sam_loss: %.4f, dae_loss: %.4f \
               ' % (epoch, iter_num, loss, loss_l, loss_u, net1_mse_loss_lab.item(), net1_mse_loss_unlab.item(),
                    net1_kl_loss_lab.item(), net1_kl_loss_unlab.item(), 
                    sam_loss_1.item() if args.use_sam else 0.0,
                    dae_loss_1.item() if args.use_dae else 0.0))

            update_ema_variables(model1, ema_model1, 0.99)

            if iter_num >= self_max_iterations:
                logging.info("Self-training reached max iterations: {}".format(self_max_iterations))
                return

        if epoch % 5 == 0:
            model1.eval()
            model2.eval()
            dice_sample = test_3d_patch.var_all_case_LA(model1, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=18, stride_z=4, data_root=train_data_path)
            dice_sample2 = test_3d_patch.var_all_case_LA(model2, num_classes=num_classes, patch_size=patch_size,
                                                         stride_xy=18, stride_z=4, data_root=train_data_path)
            mean_dice_sample = test_3d_patch.var_all_case_LA_mean(model1, model2, num_classes=num_classes,
                                                                  patch_size=patch_size, stride_xy=18, stride_z=4,
                                                                  data_root=train_data_path)

            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(self_snapshot_path, 'best_model.pth')
                torch.save(model1.state_dict(), save_mode_path)
                torch.save(model1.state_dict(), save_best_path)
                logging.info("save best model to {}".format(save_mode_path))
                logging.info("cur dice %.4f, max dice %.4f" % (dice_sample, best_dice))

            if dice_sample2 > best_dice2:
                best_dice2 = round(dice_sample2, 4)
                save_mode_path = os.path.join(self_snapshot_path,
                                              'iter_{}_dice_{}_res.pth'.format(iter_num, best_dice2))
                save_best_path = os.path.join(self_snapshot_path, 'best_model_res.pth')
                torch.save(model2.state_dict(), save_mode_path)
                torch.save(model2.state_dict(), save_best_path)
                logging.info("resnet cur dice %.4f, max dice %.4f" % (dice_sample2, best_dice2))

            if mean_dice_sample > mean_best_dice:
                mean_best_dice = round(mean_dice_sample, 4)
                save_mode_path1 = os.path.join(self_snapshot_path,
                                               'iter_{}_dice_{}_v.pth'.format(iter_num, mean_best_dice))
                save_best_path1 = os.path.join(self_snapshot_path, 'best_model_v.pth')

                save_mode_path2 = os.path.join(self_snapshot_path,
                                               'iter_{}_dice_{}_r.pth'.format(iter_num, mean_best_dice))
                save_best_path2 = os.path.join(self_snapshot_path, 'best_model_r.pth')

                torch.save(model1.state_dict(), save_mode_path1)
                torch.save(model1.state_dict(), save_best_path1)

                torch.save(model2.state_dict(), save_mode_path2)
                torch.save(model2.state_dict(), save_best_path2)

                logging.info("mean save best model to {}".format(save_mode_path1))
                logging.info("mean cur dice %.4f, max dice %.4f" % (mean_dice_sample, mean_best_dice))

            model1.train()
            model2.train()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "code/model/SDCL/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "code/model/SDCL/LA_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    
    # 如果指定了自定义预训练路径，则使用该路径
    if args.pretrain_path:
        pre_snapshot_path = args.pretrain_path
    
    print("Starting SDCL training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('code/LA_train_smi_dae.py', self_snapshot_path)
    
    # -- Pre-Training (可跳过)
    if not args.skip_pretrain:
        logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        pre_train(args, pre_snapshot_path)
    else:
        # 验证预训练权重是否存在
        required_files = [
            os.path.join(pre_snapshot_path, 'best_model.pth'),
            os.path.join(pre_snapshot_path, 'best_model_resnet.pth')
        ]
        for f in required_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"跳过预训练需要预训练权重文件: {f}")
        print(f"跳过预训练，直接使用权重: {pre_snapshot_path}")
    
    # -- DAE Pre-Training (在有标签数据上预训练DAE学习解剖先验)
    dae_snapshot_path = "code/model/SDCL/LA_{}_{}_labeled/dae_pretrain".format(args.exp, args.labelnum)
    dae_model_path = None
    
    if args.use_dae:
        if not os.path.exists(dae_snapshot_path):
            os.makedirs(dae_snapshot_path)
        
        if not args.skip_dae_pretrain:
            # 设置DAE预训练的日志
            dae_log_file = os.path.join(dae_snapshot_path, "dae_log.txt")
            dae_handler = logging.FileHandler(dae_log_file)
            dae_handler.setLevel(logging.INFO)
            dae_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
            logging.getLogger().addHandler(dae_handler)
            
            logging.info("="*60)
            logging.info("Stage: DAE Pre-training on labeled data")
            logging.info("="*60)
            
            # 执行DAE预训练
            dae_model_path = pre_train_dae(args, dae_snapshot_path)
            logging.info(f"DAE pre-training completed. Model saved to: {dae_model_path}")
        else:
            # 检查是否存在预训练的DAE模型
            dae_model_path = os.path.join(dae_snapshot_path, 'best_dae_model.pth')
            if not os.path.exists(dae_model_path):
                # 尝试使用命令行参数指定的路径
                if args.dae_model_path and os.path.exists(args.dae_model_path):
                    dae_model_path = args.dae_model_path
                    print(f"使用命令行指定的DAE模型: {dae_model_path}")
                else:
                    logging.warning(f"DAE模型未找到: {dae_model_path}，将使用随机初始化的DAE")
                    dae_model_path = None
            else:
                print(f"跳过DAE预训练，使用已有模型: {dae_model_path}")
    
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    logging.info("="*60)
    logging.info("Stage: Self-training with DAE-enhanced uncertainty estimation")
    if dae_model_path:
        logging.info(f"Using pre-trained DAE model: {dae_model_path}")
    else:
        logging.info("WARNING: No pre-trained DAE model, using random initialization")
    logging.info("="*60)
    
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    self_train(args, pre_snapshot_path, self_snapshot_path, dae_model_path=dae_model_path)
