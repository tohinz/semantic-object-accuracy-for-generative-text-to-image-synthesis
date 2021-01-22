import logging

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from miscc.config import cfg
from GlobalAttention import GlobalAttentionGeneral as ATT_NET

logger = logging.getLogger()


def stn(image, transformation_matrix, size):
    grid = torch.nn.functional.affine_grid(transformation_matrix, torch.Size(size), align_corners=False)
    out_image = torch.nn.functional.grid_sample(image, grid, align_corners=False)

    return out_image


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        # nn.functional.interpolate(scale_factor=2, mode="nearest"),
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


def channel_pool(input, kernel_size):
    b, c, h, w = input.size()
    input = input.view(b, c, h * w).permute(0, 2, 1)
    stride = c
    pooled = torch.nn.functional.max_pool1d(input, kernel_size, stride)
    pooled = pooled.permute(0, 2, 1).view(b, -1, h, w)
    assert pooled.shape[1] == 1
    return pooled.squeeze()


def merge_tensors(source, new_features, idx):
    """This method deals with the fact that some bboxes overlap each other.
    To deal with this we use the simple heuristic that the smaller bbox contains the object in the foreground.
    As such features in smaller bboxes replace the features of larger bboxes in overlapping areas"""
    if idx == 0:
        return new_features
    else:
        nz = torch.nonzero(new_features)
        source[nz[:, 0], nz[:, 1], nz[:, 2], nz[:, 3]] = new_features[nz[:, 0], nz[:, 1], nz[:, 2], nz[:, 3]]
        return source


class BBOX_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(BBOX_NET, self).__init__()
        self.c_dim = cfg.GAN.INIT_LABEL_DIM
        self.input_dim = cfg.GAN.LAYOUT_SPATIAL_DIM
        self.encode = nn.Sequential(
            # 128 * 16 x 16
            conv3x3(self.c_dim, self.c_dim // 2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 8 x 8
            conv3x3(self.c_dim // 2, self.c_dim // 4, stride=2),
            nn.BatchNorm2d(self.c_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 4 x 4
            conv3x3(self.c_dim // 4, self.c_dim // 8, stride=2),
            nn.BatchNorm2d(self.c_dim // 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 2 x 2
        )

    def forward(self, labels, transf_matr_inv, max_objects):
        label_layout = labels.new_zeros(labels.shape[0], self.c_dim, self.input_dim, self.input_dim)
        for idx in range(max_objects):
            current_label = labels[:, idx]
            current_label = current_label.view(current_label.shape[0], current_label.shape[1], 1, 1)
            current_label = current_label.repeat(1, 1, self.input_dim, self.input_dim)
            current_label = stn(current_label, transf_matr_inv[:, idx], current_label.shape)
            label_layout += current_label

        layout_encoding = self.encode(label_layout).view(labels.shape[0], -1)

        return layout_encoding


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        """
        nn.LSTM and nn.GRU will give a warning if nlayers=1 and dropout>0, saying dropout is only used
            when nlayers>1. That's okay.
        """
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, feat_dim):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.feat_dim = feat_dim
        else:
            self.feat_dim = cfg.TEXT.EMBEDDING_DIM

        model = models.inception_v3(init_weights=False)
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        state_dict = torch.hub.load_state_dict_from_url(url, model_dir='models/hub')
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad = False
        logger.info('Load pretrained model from %s', url)
        # logger.info(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.feat_dim)
        self.emb_cnn_code = nn.Linear(2048, self.feat_dim)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        # x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear')
        x = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code


# ############## G networks ###################
class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.TEXT_CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.zeros_like(std).normal_()
        eps = Variable(eps)
        mult = eps.mul(std)
        added = mult.add(mu)
        return added

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self):
        super(INIT_STAGE_G, self).__init__()
        self.gen_feat_dim = cfg.GAN.GEN_FEAT_DIM * 16
        self.define_module()

    def define_module(self):
        self.bbox_net = BBOX_NET()

        layout_cond_dim = cfg.GAN.INIT_LABEL_DIM // 8 * (cfg.GAN.LAYOUT_SPATIAL_DIM // 8)**2
        cond_inp_dim = cfg.GAN.TEXT_CONDITION_DIM + cfg.GAN.GLOBAL_Z_DIM + layout_cond_dim
        self.fc = nn.Sequential(
            nn.Linear(cond_inp_dim, self.gen_feat_dim * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.gen_feat_dim * 4 * 4 * 2),
            GLU()
        )

        # local pathway
        label_inp_dim = cfg.GAN.TEXT_CONDITION_DIM + cfg.TEXT.CLASSES_NUM + cfg.GAN.LOCAL_Z_DIM
        self.label = nn.Sequential(
            nn.Linear(label_inp_dim, cfg.GAN.INIT_LABEL_DIM, bias=False),
            nn.BatchNorm1d(cfg.GAN.INIT_LABEL_DIM),
            nn.ReLU(inplace=True)
        )
        self.local1 = upBlock(cfg.GAN.INIT_LABEL_DIM, self.gen_feat_dim // 2)
        self.local2 = upBlock(self.gen_feat_dim // 2, self.gen_feat_dim // 4)

        self.upsample1 = upBlock(self.gen_feat_dim, self.gen_feat_dim // 2)
        self.upsample2 = upBlock(self.gen_feat_dim // 2, self.gen_feat_dim // 4)
        self.upsample3 = upBlock(self.gen_feat_dim // 2, self.gen_feat_dim // 8)
        self.upsample4 = upBlock(self.gen_feat_dim // 8, self.gen_feat_dim // 16)

    def forward(self, z_code, local_noise, c_code, transf_matrices_inv, label_one_hot, max_objects, op=True):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        local_labels = z_code.new_zeros(z_code.shape[0], max_objects, cfg.GAN.INIT_LABEL_DIM)

        # object pathway
        h_code_locals = z_code.new_zeros(z_code.shape[0], self.gen_feat_dim // 4, 16, 16)

        if op:
            for idx in range(max_objects):
                current_label = self.label(torch.cat((c_code, label_one_hot[:, idx], local_noise), 1))
                local_labels[:, idx] = current_label
                current_label = current_label.view(current_label.shape[0], cfg.GAN.INIT_LABEL_DIM, 1, 1)
                current_label = current_label.repeat(1, 1, 4, 4)
                h_code_local = self.local1(current_label)
                h_code_local = self.local2(h_code_local)
                h_code_local = stn(h_code_local, transf_matrices_inv[:, idx], h_code_local.shape)
                h_code_locals = merge_tensors(h_code_locals, h_code_local, idx)

        bbox_code = self.bbox_net(local_labels, transf_matrices_inv, max_objects)
        c_z_code = torch.cat((c_code, z_code, bbox_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gen_feat_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)

        # combine local and global pathways
        out_code = torch.cat((out_code, h_code_locals), 1)

        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64


class NEXT_STAGE_G(nn.Module):
    def __init__(self):
        super(NEXT_STAGE_G, self).__init__()
        self.gen_feat_dim = cfg.GAN.GEN_FEAT_DIM
        self.text_emb_dim = cfg.TEXT.EMBEDDING_DIM
        self.label_dim = cfg.GAN.NEXT_LABEL_DIM
        self.num_residual = cfg.GAN.RESIDUAL_NUM
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.RESIDUAL_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        self.att = ATT_NET(self.gen_feat_dim, self.text_emb_dim)
        self.residual = self._make_layer(ResBlock, self.gen_feat_dim * 2)
        self.upsample = upBlock(self.gen_feat_dim * 3, self.gen_feat_dim)

        # local pathway
        label_input_dim = cfg.GAN.TEXT_CONDITION_DIM + cfg.TEXT.CLASSES_NUM  # no noise anymore
        self.label = nn.Sequential(
            nn.Linear(label_input_dim, self.label_dim, bias=False),
            nn.BatchNorm1d(self.label_dim),
            nn.ReLU(True)
        )

        self.local1 = upBlock(self.label_dim + self.gen_feat_dim, self.gen_feat_dim * 2)
        self.local2 = upBlock(self.gen_feat_dim * 2, self.gen_feat_dim)

    def forward(self, h_code, c_code, word_embs, mask, transf_matrices, transf_matrices_inv, label_one_hot,
                max_objects, op=True):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        _hw = h_code.shape[2]
        self.att.applyMask(mask)
        c_code_att, att = self.att(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code_att), 1)
        out_code = self.residual(h_c_code)

        # object pathways
        h_code_locals = h_code.new_zeros(h_code.shape[0], self.gen_feat_dim, _hw, _hw)
        if op:
            for idx in range(max_objects):
                current_label = self.label(torch.cat((c_code, label_one_hot[:, idx]), 1))
                current_label = current_label.view(h_code.shape[0], self.label_dim, 1, 1)
                current_label = current_label.repeat(1, 1, _hw//4, _hw//4)
                current_patch = stn(h_code, transf_matrices[:, idx], (h_code.shape[0], h_code.shape[1], _hw//4, _hw//4))
                # logger.info(current_label.shape)
                # logger.info(current_patch.shape)
                current_input = torch.cat((current_patch, current_label), 1)
                # logger.info(current_input.shape)
                h_code_local = self.local1(current_input)
                h_code_local = self.local2(h_code_local)
                h_code_local = stn(h_code_local, transf_matrices_inv[:, idx], h_code_locals.shape)
                h_code_locals = merge_tensors(h_code_locals, h_code_local, idx)

        out_code = torch.cat((out_code, h_code_locals), 1)

        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code, att


class GET_IMAGE_G(nn.Module):
    def __init__(self):
        super(GET_IMAGE_G, self).__init__()
        self.img = nn.Sequential(
            conv3x3(cfg.GAN.GEN_FEAT_DIM, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G()
            self.img_net1 = GET_IMAGE_G()
        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G()
            self.img_net2 = GET_IMAGE_G()
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G()
            self.img_net3 = GET_IMAGE_G()

    def forward(self, z_code, local_noise, sent_emb, word_embs, mask, transf_matrices, transf_matrices_inv,
                label_one_hot, max_objects, op=[True, True, True]):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)

        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, local_noise, c_code, transf_matrices_inv, label_one_hot, max_objects, op[0])
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = self.h_net2(h_code1, c_code, word_embs, mask, transf_matrices,
                                        transf_matrices_inv, label_one_hot, max_objects, op[1])
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = self.h_net3(h_code2, c_code, word_embs, mask, transf_matrices,
                                        transf_matrices_inv, label_one_hot, max_objects, op[2])
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)

        return fake_imgs, att_maps, mu, logvar


# ############## D networks ##########################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)

        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET64, self).__init__()
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(cfg.GAN.DISC_FEAT_DIM, cfg.TEXT.EMBEDDING_DIM, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(cfg.GAN.DISC_FEAT_DIM, cfg.TEXT.EMBEDDING_DIM, bcondition=True)
        self.define_module()

    def define_module(self):
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # global pathway
        # --> state size. ndf x in_size/2 x in_size/2
        self.conv1 = nn.Conv2d(3, cfg.GAN.DISC_FEAT_DIM, 4, 2, 1, bias=False)
        # --> state size 2ndf x x in_size/4 x in_size/4
        self.conv2 = nn.Conv2d(cfg.GAN.DISC_FEAT_DIM, cfg.GAN.DISC_FEAT_DIM * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM * 2)
        # --> state size 4ndf x in_size/8 x in_size/8
        self.conv3 = nn.Conv2d(cfg.GAN.DISC_FEAT_DIM * 4, cfg.GAN.DISC_FEAT_DIM * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM * 4)
        # --> state size 8ndf x in_size/16 x in_size/16
        self.conv4 = nn.Conv2d(cfg.GAN.DISC_FEAT_DIM * 4, cfg.GAN.DISC_FEAT_DIM * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM * 8)

        # object pathway
        self.local = nn.Sequential(
            nn.Conv2d(3 + cfg.TEXT.CLASSES_NUM, cfg.GAN.DISC_FEAT_DIM * 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, image, label, transf_matrices, transf_matrices_inv, max_objects):
        # object pathway
        h_code_locals = image.new_zeros(image.shape[0], cfg.GAN.DISC_FEAT_DIM * 2, 16, 16, dtype=torch.float)

        for idx in range(max_objects):
            current_label = label[:, idx].view(label.shape[0], cfg.TEXT.CLASSES_NUM, 1, 1)
            current_label = current_label.repeat(1, 1, 16, 16)
            h_code_local = stn(image, transf_matrices[:, idx], (image.shape[0], image.shape[1], 16, 16))
            h_code_local = torch.cat((h_code_local, current_label), 1)
            h_code_local = self.local(h_code_local)
            h_code_local = stn(h_code_local, transf_matrices_inv[:, idx],
                               (h_code_local.shape[0], h_code_local.shape[1], 16, 16))
            h_code_locals = merge_tensors(h_code_locals, h_code_local, idx)

        h_code = self.conv1(image)
        h_code = self.act(h_code)
        h_code = self.conv2(h_code)
        h_code = self.bn2(h_code)
        h_code = self.act(h_code)

        h_code = torch.cat((h_code, h_code_locals), 1)

        h_code = self.conv3(h_code)
        h_code = self.bn3(h_code)
        h_code = self.act(h_code)

        h_code = self.conv4(h_code)
        h_code = self.bn4(h_code)
        x_code4 = self.act(h_code)

        return x_code4


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET128, self).__init__()
        self.img_code_s32 = downBlock(cfg.GAN.DISC_FEAT_DIM * 8, cfg.GAN.DISC_FEAT_DIM * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(cfg.GAN.DISC_FEAT_DIM * 16, cfg.GAN.DISC_FEAT_DIM * 8)
        self.encode_img = nn.Sequential(
            # --> state size. ndf x in_size/2 x in_size/2
            nn.Conv2d(3, cfg.GAN.DISC_FEAT_DIM, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # --> state size 2ndf x x in_size/4 x in_size/4
            nn.Conv2d(cfg.GAN.DISC_FEAT_DIM, cfg.GAN.DISC_FEAT_DIM * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encode_final = nn.Sequential(
            nn.Conv2d(cfg.GAN.DISC_FEAT_DIM * 4, cfg.GAN.DISC_FEAT_DIM * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # --> state size 8ndf x in_size/16 x in_size/16
            nn.Conv2d(cfg.GAN.DISC_FEAT_DIM * 4, cfg.GAN.DISC_FEAT_DIM * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        #
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(cfg.GAN.DISC_FEAT_DIM, cfg.TEXT.EMBEDDING_DIM, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(cfg.GAN.DISC_FEAT_DIM, cfg.TEXT.EMBEDDING_DIM, bcondition=True)

        self.local = nn.Sequential(
            nn.Conv2d(3 + cfg.TEXT.CLASSES_NUM, cfg.GAN.DISC_FEAT_DIM, 4, 1, 1, bias=False),
            nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.GAN.DISC_FEAT_DIM, cfg.GAN.DISC_FEAT_DIM * 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image, label, transf_matrices, transf_matrices_inv, max_objects):
        # object pathway
        h_code_locals = image.new_zeros(image.shape[0], cfg.GAN.DISC_FEAT_DIM * 2, 32, 32, dtype=torch.float)

        for idx in range(max_objects):
            current_label = label[:, idx].view(label.shape[0], cfg.TEXT.CLASSES_NUM, 1, 1)
            current_label = current_label.repeat(1, 1, 32, 32)
            h_code_local = stn(image, transf_matrices[:, idx], (image.shape[0], image.shape[1], 32, 32))
            h_code_local = torch.cat((h_code_local, current_label), 1)
            h_code_local = self.local(h_code_local)
            h_code_local = stn(h_code_local, transf_matrices_inv[:, idx],
                               (h_code_local.shape[0], h_code_local.shape[1], 32, 32))
            h_code_locals = merge_tensors(h_code_locals, h_code_local, idx)

        x_code_32 = self.encode_img(image)  # 32 x 32 x df*2
        x_code_32 = torch.cat((x_code_32, h_code_locals), 1)  # 32 x 32 x df*4

        x_code8 = self.encode_final(x_code_32)  # 8 x 8 x 8df
        x_code4 = self.img_code_s32(x_code8)  # 4 x 4 x 16df
        x_code4 = self.img_code_s32_1(x_code4)  # 4 x 4 x 8df
        return x_code4


# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET256, self).__init__()
        self.img_code_s16 = encode_image_by_16times(cfg.GAN.DISC_FEAT_DIM)
        self.encode_img = nn.Sequential(
            # --> state size. ndf x in_size/2 x in_size/2
            nn.Conv2d(3, cfg.GAN.DISC_FEAT_DIM, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # --> state size 2ndf x x in_size/4 x in_size/4
            nn.Conv2d(cfg.GAN.DISC_FEAT_DIM, cfg.GAN.DISC_FEAT_DIM * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encode_final = nn.Sequential(
            nn.Conv2d(cfg.GAN.DISC_FEAT_DIM * 4, cfg.GAN.DISC_FEAT_DIM * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # --> state size 8ndf x in_size/16 x in_size/16
            nn.Conv2d(cfg.GAN.DISC_FEAT_DIM * 4, cfg.GAN.DISC_FEAT_DIM * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_code_s32 = downBlock(cfg.GAN.DISC_FEAT_DIM * 8, cfg.GAN.DISC_FEAT_DIM * 16)
        self.img_code_s64 = downBlock(cfg.GAN.DISC_FEAT_DIM * 16, cfg.GAN.DISC_FEAT_DIM * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(cfg.GAN.DISC_FEAT_DIM * 32, cfg.GAN.DISC_FEAT_DIM * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(cfg.GAN.DISC_FEAT_DIM * 16, cfg.GAN.DISC_FEAT_DIM * 8)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(cfg.GAN.DISC_FEAT_DIM, cfg.TEXT.EMBEDDING_DIM, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(cfg.GAN.DISC_FEAT_DIM, cfg.TEXT.EMBEDDING_DIM, bcondition=True)

        self.local = nn.Sequential(
            nn.Conv2d(3 + cfg.TEXT.CLASSES_NUM, cfg.GAN.DISC_FEAT_DIM, 4, 1, 1, bias=False),
            nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.GAN.DISC_FEAT_DIM, cfg.GAN.DISC_FEAT_DIM * 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(cfg.GAN.DISC_FEAT_DIM * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image, label, transf_matrices, transf_matrices_inv, max_objects):
        # object pathway
        h_code_locals = image.new_zeros(image.shape[0], cfg.GAN.DISC_FEAT_DIM * 2, 64, 64, dtype=torch.float)

        for idx in range(max_objects):
            current_label = label[:, idx].view(label.shape[0], cfg.TEXT.CLASSES_NUM, 1, 1)
            current_label = current_label.repeat(1, 1, 64, 64)
            h_code_local = stn(image, transf_matrices[:, idx], (image.shape[0], image.shape[1], 64, 64))
            h_code_local = torch.cat((h_code_local, current_label), 1)
            h_code_local = self.local(h_code_local)
            h_code_local = stn(h_code_local, transf_matrices_inv[:, idx],
                               (h_code_local.shape[0], h_code_local.shape[1], 64, 64))
            h_code_locals = merge_tensors(h_code_locals, h_code_local, idx)

        x_code_64 = self.encode_img(image)
        x_code_64 = torch.cat((x_code_64, h_code_locals), 1)

        x_code16 = self.encode_final(x_code_64)
        x_code8 = self.img_code_s32(x_code16)
        x_code4 = self.img_code_s64(x_code8)
        x_code4 = self.img_code_s64_1(x_code4)
        x_code4 = self.img_code_s64_2(x_code4)

        return x_code4
