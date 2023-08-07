from model import Generator
from model import Discriminator
import torch
import torch.nn.functional as F
import os
import time
import datetime
import scipy.io as scio


class Solver(object):
    """Solver for training"""

    def __init__(self, data_loader, config, all_grid_item):
        """Initialize configurations."""
        self.base_path = config.base_path
        # Data loader.
        self.data_loader = data_loader
        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_reg = config.lambda_reg
        self.lambda_rec = config.lambda_rec
        self.lambda_idn = config.lambda_idn
        self.lambda_gp = config.lambda_gp
        self.lambda_para = config.lambda_para
        self.lambda_sym = config.lambda_sym

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.resume_iters = config.resume_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.all_grid_item = all_grid_item

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""

        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num, self.all_grid_item)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num, self.all_grid_item)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr)
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model, file=open(os.path.join(self.base_path, 'config.txt'), 'a'))
        print("The number of parameters in " + name + " : {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.base_path, self.model_save_dir, '{}-G.pth'.format(resume_iters))
        D_path = os.path.join(self.base_path, self.model_save_dir, '{}-D.pth'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path), strict=True)
        self.D.load_state_dict(torch.load(D_path), strict=True)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from tensorboardX import SummaryWriter
        self.logger = SummaryWriter(os.path.join(self.base_path, self.log_dir))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - lambda)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.mean(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - self.lambda_para) ** 2)

    def create_labels(self, c_org, c_dim):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim + 2):
            c_trg = c_org.clone()
            if i < 20:  # Set expression attribute to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in range(20):
                    if j != i:
                        c_trg[:, j] = 0
            elif i == 20:
                c_trg[:, 20] = 1  # Reverse gender attribute.
                c_trg[:, 21] = 0  # Reverse gender attribute.
            elif i == 21:
                c_trg[:, 20] = 0  # Reverse gender attribute.
                c_trg[:, 21] = 1  # Reverse gender attribute.
            elif i == 22:
                c_trg[:, 22] = 1  # Reverse symmetry attribute.
                c_trg[:, 23] = 0  # Reverse symmetry attribute.
            elif i == 23:
                c_trg[:, 22] = 0  # Reverse symmetry attribute.
                c_trg[:, 23] = 1  # Reverse symmetry attribute.
            # elif i < 24:
            #     c_trg[:, i] = (c_trg[:, i] == 0) # Reverse gender and symmetry attribute.
            elif i == 24:
                c_trg[:, 24] = -1  # set age attribute
            elif i == 25:
                c_trg[:, 24] = 0  # set age attribute
            elif i == 26:
                c_trg[:, 24] = 1  # set age attribute
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary cross entropy loss for expression and gender."""
        return F.binary_cross_entropy_with_logits(logit, target)

    def regression_loss(self, logit, target):
        """Compute mse loss for age."""
        return F.mse_loss(logit, target)
        # return F.mse_loss(logit, target, reduction='sum') / logit.size(0)

    def train(self):
        """Training 3DFaceGAN"""
        data_loader = self.data_loader

        # Fetch fixed inputs for validation.
        data_iter = iter(data_loader)
        x_fixed, c_org, rot_angle_fixed_denorm, scale_norm = next(data_iter)
        x_fixed = torch.bmm(x_fixed/(scale_norm.unsqueeze(2).repeat(1,x_fixed.size(1),x_fixed.size(2))), rot_angle_fixed_denorm.permute(0, 2, 1))
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim)  # create_false_labels: one validation image for view
        rot_angle_fixed = torch.eye(3).unsqueeze(0).repeat(x_fixed.size(0), 1, 1).to(self.device)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org, rot_angle, _ = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org, rot_angle, _ = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label_org.clone()
            c_trg = label_trg.clone()

            x_real = x_real.to(self.device)  # Input images.
            c_org = c_org.to(self.device)  # Original domain labels.
            c_trg = c_trg.to(self.device)  # Target domain labels.
            label_org = label_org.to(self.device)  # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)  # Labels for computing classification loss.
            rot_angle = rot_angle.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls[:, :-1], label_org[:, :-1])
            d_loss_reg = self.regression_loss(out_cls[:, -1], label_org[:, -1])

            # Compute loss with fake images.
            x_fake, _ = self.G(x_real, c_trg, rot_angle)
            out_src, _ = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_reg * d_loss_reg + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_reg'] = d_loss_reg.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_real2 = torch.cat([x_real, x_real], dim=0)
                c_rg2 = torch.cat([c_trg, c_org], dim=0)
                rot_angle2 = torch.cat([rot_angle, rot_angle], dim=0)
                x_fake_and_real, sym_error = self.G(x_real2, c_rg2, rot_angle2)
                x_fake = x_fake_and_real[:x_real.size(0), :, :]
                x_real_i = x_fake_and_real[x_real.size(0):, :, :]

                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_sym = torch.mean(torch.abs(sym_error))
                g_loss_cls = self.classification_loss(out_cls[:, :-1], label_trg[:, :-1])
                g_loss_reg = self.regression_loss(out_cls[:, -1], label_trg[:, -1])
                g_loss_idn = torch.mean(torch.abs(x_real - x_real_i)) #reconstruction loss

                # Target-to-original domain.
                x_reconst, _ = self.G(x_fake, c_org, rot_angle)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst)) #cycle loss

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_idn * g_loss_idn + self.lambda_cls * g_loss_cls + self.lambda_reg * g_loss_reg + self.lambda_sym * g_loss_sym
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_idn'] = g_loss_idn.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_reg'] = g_loss_reg.item()
                loss['G/loss_sym'] = g_loss_sym.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.add_scalar(tag, value, i + 1)
            # Validation.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = x_fixed.clone().cpu().unsqueeze(0)
                    for c_fixed in c_fixed_list:
                        x_output_fake, _ = self.G(x_fixed, c_fixed, rot_angle_fixed)
                        x_fake_list = torch.cat((x_fake_list, x_output_fake.cpu().unsqueeze(0)), dim=0)
                    sample_path = os.path.join(self.base_path, self.sample_dir, '{}.mat'.format(i + 1))
                    scio.savemat(sample_path, {'shape': x_fake_list.numpy()})
                    print('Saved generated data into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.base_path, self.model_save_dir, '{}-G.pth'.format(i + 1))
                D_path = os.path.join(self.base_path, self.model_save_dir, '{}-D.pth'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints to {}...'.format(os.path.join(self.base_path, self.model_save_dir)))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
