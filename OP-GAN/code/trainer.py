from __future__ import print_function

import logging

from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import glob

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, count_learnable_params, DataParallelPassThrough
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_NET
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss

logger = logging.getLogger()


# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, resume):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.resume = resume

        self.n_gpu = torch.cuda.device_count()

        self.n_words = n_words
        self.ixtoword = ixtoword

        if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
            self.data_loader = data_loader
            self.num_batches = 0
            self.subset_lengths = []
            for _idx in range(len(self.data_loader)):
                self.num_batches += len(self.data_loader[_idx])
                self.subset_lengths.append(len(self.data_loader[_idx]))
        else:
            self.data_loader = data_loader
            self.num_batches = len(self.data_loader)

        if cfg.CUDA:
            torch.cuda.set_device(cfg.DEVICE)
            cudnn.benchmark = True

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            raise Exception('Error: no pretrained text encoder')

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        logger.info('Load image encoder from: %s', img_encoder_path)
        image_encoder.eval()

        text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        logger.info('Load text encoder from: %s', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        from model import D_NET64, D_NET128, D_NET256
        netG = G_NET()
        if cfg.TREE.BRANCH_NUM > 0:
            netsD.append(D_NET64())
        if cfg.TREE.BRANCH_NUM > 1:
            netsD.append(D_NET128())
        if cfg.TREE.BRANCH_NUM > 2:
            netsD.append(D_NET256())

        netG.apply(weights_init)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
        logger.info('# of params in netG: %s' % count_learnable_params(netG))
        logger.info('# of netsD: %s', len(netsD))
        logger.info('# of params in netsD: %s' % [count_learnable_params(netD) for netD in netsD])
        epoch = 0

        if self.resume:
            checkpoint_list = sorted([ckpt for ckpt in glob.glob(self.model_dir + "/" + '*.pth')])
            latest_checkpoint = checkpoint_list[-1]
            state_dict = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)

            netG.load_state_dict(state_dict["netG"])
            for i in range(len(netsD)):
                netsD[i].load_state_dict(state_dict["netD"][i])
            epoch = int(latest_checkpoint[-8:-4]) + 1
            logger.info("Resuming training from checkpoint {} at epoch {}.".format(latest_checkpoint, epoch))

        #
        if cfg.TRAIN.NET_G != '':
            state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            logger.info('Load G from: %s', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    logger.info('Load D from: %s', Dname)
                    state_dict = torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            text_encoder.to(cfg.DEVICE)
            image_encoder.to(cfg.DEVICE)
            netG.to(cfg.DEVICE)
            if self.n_gpu > 1:
                netG = DataParallelPassThrough(netG, )
            for i in range(len(netsD)):
                netsD[i].to(cfg.DEVICE)
                if self.n_gpu > 1:
                    netsD[i] = DataParallelPassThrough(netsD[i], )
        return [text_encoder, image_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        if self.resume:
            checkpoint_list = sorted([ckpt for ckpt in glob.glob(self.model_dir + "/" + '*.pth')])
            latest_checkpoint = checkpoint_list[-1]
            state_dict = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            optimizerG.load_state_dict(state_dict["optimG"])

            for i in range(len(netsD)):
                optimizersD[i].load_state_dict(state_dict["optimD"][i])

        return optimizerG, optimizersD

    def prepare_labels(self):
        if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
            batch_sizes = self.batch_size
            real_labels, fake_labels, match_labels = [], [], []
            for batch_size in batch_sizes:
                real_labels.append(Variable(torch.FloatTensor(batch_size).fill_(1).to(cfg.DEVICE).detach()))
                fake_labels.append(Variable(torch.FloatTensor(batch_size).fill_(0).to(cfg.DEVICE).detach()))
                match_labels.append(Variable(torch.LongTensor(range(batch_size)).to(cfg.DEVICE).detach()))
        else:
            batch_size = self.batch_size[0]
            real_labels = Variable(torch.FloatTensor(batch_size).fill_(1).to(cfg.DEVICE).detach())
            fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0).to(cfg.DEVICE).detach())
            match_labels = Variable(torch.LongTensor(range(batch_size)).to(cfg.DEVICE).detach())

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, optimG, optimsD, epoch, max_to_keep=5, interval=5):
        netDs_state_dicts = []
        optimDs_state_dicts = []
        for i in range(len(netsD)):
            netD = netsD[i]
            optimD = optimsD[i]
            netDs_state_dicts.append(netD.state_dict())
            optimDs_state_dicts.append(optimD.state_dict())

        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        checkpoint = {
            'epoch': epoch,
            'netG': netG.state_dict(),
            'optimG': optimG.state_dict(),
            'netD': netDs_state_dicts,
            'optimD': optimDs_state_dicts}
        torch.save(checkpoint, "{}/checkpoint_{:04}.pth".format(self.model_dir, epoch))
        logger.info('Save G/D models')

        load_params(netG, backup_para)

        if max_to_keep is not None and max_to_keep > 0:
            checkpoint_list_all = sorted([ckpt for ckpt in glob.glob(self.model_dir + "/" + '*.pth')])
            checkpoint_list = []
            checkpoint_list_tmp = []

            for ckpt in checkpoint_list_all:
                ckpt_epoch = int(ckpt[-8:-4])
                if ckpt_epoch % interval == 0:
                    checkpoint_list.append(ckpt)
                else:
                    checkpoint_list_tmp.append(ckpt)

            while len(checkpoint_list) > max_to_keep:
                os.remove(checkpoint_list[0])
                checkpoint_list = checkpoint_list[1:]

            ckpt_tmp = len(checkpoint_list_tmp)
            for idx in range(ckpt_tmp-1):
                os.remove(checkpoint_list_tmp[idx])

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, transf_matrices_inv, label_one_hot, local_noise,
                         transf_matrices, max_objects, subset_idx, name='current'):
        # Save images
        inputs = (noise, local_noise, sent_emb, words_embs, mask, transf_matrices, transf_matrices_inv,
                  label_one_hot, max_objects)
        fake_imgs, attention_maps, _, _ = netG(*inputs)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = build_super_images(img, captions, self.ixtoword, attn_maps, att_sze, lr_imgs=lr_img,
                                            batch_size=self.batch_size[0])
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png' % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
            _, _, att_maps = words_loss(region_features.detach(), words_embs.detach(),
                                        None, cap_lens, None, self.batch_size[subset_idx])
        else:
            _, _, att_maps = words_loss(region_features.detach(), words_embs.detach(),
                                        None, cap_lens, None, self.batch_size[0])
        img_set, _ = build_super_images(fake_imgs[i].detach().cpu(), captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png' % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
            batch_sizes = self.batch_size
            noise, local_noise, fixed_noise = [], [], []
            for batch_size in batch_sizes:
                noise.append(Variable(torch.FloatTensor(batch_size, cfg.GAN.GLOBAL_Z_DIM)).to(cfg.DEVICE))
                local_noise.append(Variable(torch.FloatTensor(batch_size, cfg.GAN.LOCAL_Z_DIM)).to(cfg.DEVICE))
                fixed_noise.append(Variable(torch.FloatTensor(batch_size, cfg.GAN.GLOBAL_Z_DIM).normal_(0, 1)).to(cfg.DEVICE))
        else:
            batch_size = self.batch_size[0]
            noise = Variable(torch.FloatTensor(batch_size, cfg.GAN.GLOBAL_Z_DIM)).to(cfg.DEVICE)
            local_noise = Variable(torch.FloatTensor(batch_size, cfg.GAN.LOCAL_Z_DIM)).to(cfg.DEVICE)
            fixed_noise = Variable(torch.FloatTensor(batch_size, cfg.GAN.GLOBAL_Z_DIM).normal_(0, 1)).to(cfg.DEVICE)

        for epoch in range(start_epoch, self.max_epoch):
            logger.info("Epoch nb: %s" % epoch)
            gen_iterations = 0
            if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
                data_iter = []
                for _idx in range(len(self.data_loader)):
                    data_iter.append(iter(self.data_loader[_idx]))
                total_batches_left = sum([len(self.data_loader[i]) for i in range(len(self.data_loader))])
                current_probability = [len(self.data_loader[i]) for i in range(len(self.data_loader))]
                current_probability_percent = [current_probability[i] / float(total_batches_left) for i in
                                               range(len(current_probability))]
            else:
                data_iter = iter(self.data_loader)

            _dataset = tqdm(range(self.num_batches))
            for step in _dataset:
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
                    subset_idx = np.random.choice(range(len(self.data_loader)), size=None,
                                                  p=current_probability_percent)
                    total_batches_left -= 1
                    if total_batches_left > 0:
                        current_probability[subset_idx] -= 1
                        current_probability_percent = [current_probability[i] / float(total_batches_left) for i in
                                                       range(len(current_probability))]

                    max_objects = subset_idx
                    data = data_iter[subset_idx].next()
                else:
                    data = data_iter.next()
                    max_objects = 3
                _dataset.set_description('Obj-{}'.format(max_objects))

                imgs, captions, cap_lens, class_ids, keys, transformation_matrices, label_one_hot = prepare_data(data)
                transf_matrices = transformation_matrices[0]
                transf_matrices_inv = transformation_matrices[1]

                with torch.no_grad():
                    if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
                        hidden = text_encoder.init_hidden(batch_sizes[subset_idx])
                    else:
                        hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0).bool()
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
                    noise[subset_idx].data.normal_(0, 1)
                    local_noise[subset_idx].data.normal_(0, 1)
                    inputs = (noise[subset_idx], local_noise[subset_idx], sent_emb, words_embs, mask, transf_matrices,
                              transf_matrices_inv, label_one_hot, max_objects)
                else:
                    noise.data.normal_(0, 1)
                    local_noise.data.normal_(0, 1)
                    inputs = (noise, local_noise, sent_emb, words_embs, mask, transf_matrices, transf_matrices_inv,
                              label_one_hot, max_objects)

                inputs = tuple((inp.to(cfg.DEVICE) if isinstance(inp, torch.Tensor) else inp) for inp in inputs)
                fake_imgs, _, mu, logvar = netG(*inputs)

                #######################################################
                # (3) Update D network
                ######################################################
                # errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
                        errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                                  sent_emb, real_labels[subset_idx], fake_labels[subset_idx],
                                                  local_labels=label_one_hot, transf_matrices=transf_matrices,
                                                  transf_matrices_inv=transf_matrices_inv, cfg=cfg,
                                                  max_objects=max_objects)
                    else:
                        errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                                  sent_emb, real_labels, fake_labels,
                                                  local_labels=label_one_hot, transf_matrices=transf_matrices,
                                                  transf_matrices_inv=transf_matrices_inv, cfg=cfg,
                                                  max_objects=max_objects)

                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                # step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                netG.zero_grad()
                if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
                    errG_total = \
                        generator_loss(netsD, image_encoder, fake_imgs, real_labels[subset_idx],
                                       words_embs, sent_emb, match_labels[subset_idx], cap_lens, class_ids,
                                       local_labels=label_one_hot, transf_matrices=transf_matrices,
                                       transf_matrices_inv=transf_matrices_inv, max_objects=max_objects)
                else:
                    errG_total = \
                        generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                       words_embs, sent_emb, match_labels, cap_lens, class_ids,
                                       local_labels=label_one_hot, transf_matrices=transf_matrices,
                                       transf_matrices_inv=transf_matrices_inv, max_objects=max_objects)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(p.data, alpha=0.001)

                if cfg.TRAIN.EMPTY_CACHE:
                    torch.cuda.empty_cache()

                # save images
                if (
                        2 * gen_iterations == self.num_batches
                        or 2 * gen_iterations + 1 == self.num_batches
                        or gen_iterations + 1 == self.num_batches
                ):
                    logger.info('Saving images...')
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
                        self.save_img_results(netG, fixed_noise[subset_idx], sent_emb,
                                              words_embs, mask, image_encoder,
                                              captions, cap_lens, epoch, transf_matrices_inv,
                                              label_one_hot, local_noise[subset_idx], transf_matrices,
                                          max_objects, subset_idx, name='average')
                    else:
                        self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, transf_matrices_inv,
                                          label_one_hot, local_noise, transf_matrices,
                                          max_objects, None, name='average')
                    load_params(netG, backup_para)

            self.save_model(netG, avg_param_G, netsD, optimizerG, optimizersD, epoch)
        self.save_model(netG, avg_param_G, netsD, optimizerG, optimizersD, epoch)

    def sampling(self, split_dir, num_samples=30000):
        if cfg.TRAIN.NET_G == '':
            logger.error('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.to(cfg.DEVICE)
            netG.eval()
            #
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            text_encoder = text_encoder.to(cfg.DEVICE)
            text_encoder.eval()
            logger.info('Loaded text encoder from: %s', cfg.TRAIN.NET_E)

            batch_size = self.batch_size[0]
            nz = cfg.GAN.GLOBAL_Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz)).to(cfg.DEVICE)
            local_noise = Variable(torch.FloatTensor(batch_size, cfg.GAN.LOCAL_Z_DIM)).to(cfg.DEVICE)

            model_dir = cfg.TRAIN.NET_G
            state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict["netG"])
            max_objects = 10
            logger.info('Load G from: %s', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')].split("/")[-1]
            save_dir = '%s/%s/%s' % ("../output", s_tmp, split_dir)
            mkdir_p(save_dir)
            logger.info("Saving images to: {}".format(save_dir))

            number_batches = num_samples // batch_size
            if number_batches < 1:
                number_batches = 1

            data_iter = iter(self.data_loader)

            for step in tqdm(range(number_batches)):
                data = data_iter.next()

                imgs, captions, cap_lens, class_ids, keys, transformation_matrices, label_one_hot, _ = prepare_data(
                    data, eval=True)

                transf_matrices = transformation_matrices[0]
                transf_matrices_inv = transformation_matrices[1]

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                local_noise.data.normal_(0, 1)
                inputs = (noise, local_noise, sent_emb, words_embs, mask, transf_matrices, transf_matrices_inv, label_one_hot, max_objects)
                inputs = tuple((inp.to(cfg.DEVICE) if isinstance(inp, torch.Tensor) else inp) for inp in inputs)

                with torch.no_grad():
                    fake_imgs, _, mu, logvar = netG(*inputs)
                for batch_idx, j in enumerate(range(batch_size)):
                    s_tmp = '%s/%s' % (save_dir, keys[j])
                    folder = s_tmp[:s_tmp.rfind('/')]
                    if not os.path.isdir(folder):
                        logger.info('Make a new folder: %s', folder)
                        mkdir_p(folder)
                    k = -1
                    # for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_s%d.png' % (s_tmp, step*batch_size+batch_idx)
                    im.save(fullpath)
