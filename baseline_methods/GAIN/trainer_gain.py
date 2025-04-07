import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from utils import renormalization


class Trainer():
    def __init__(self,
                 data_loader,
                 norm_parameters,
                 generator,
                 discriminator,
                 optim_g,
                 optim_d,
                 hint_rate,
                 coef,
                 n_epoches,
                 ):

        self.device = self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.norm_parameters = norm_parameters
        self.generator = nn.DataParallel(generator).to(self.device)
        self.discriminator = nn.DataParallel(discriminator).to(self.device)
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.hint_rate = hint_rate
        self.coef = coef
        self.n_epoches = n_epoches

    def _binary_sampler(self, shape, rate):
        b = torch.rand(shape).to(self.device)
        return torch.where(b < rate, 1.0, 0.0)

    def _uniform_sampler(self, shape):
        return torch.rand(shape).to(self.device) / 100.0

    def _fool_loss(self, prob_d, m):
        return -torch.mean((1.0 - m) * torch.log(prob_d + 1e-8))

    def _discr_loss(self, prob_d, m):
        return -torch.mean((1.0 - m) * torch.log(1.0 - prob_d + 1e-8) + m * torch.log(prob_d + 1e-8))

    def _prediction_loss(self, pred_y, real_y, m):
        diff = pred_y - real_y
        return torch.sum(torch.square(diff) * m) / torch.sum(m)

    def init_weights(self):
        for layer in self.generator.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        for layer in self.discriminator.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)


    def train(self):
        loss_d_list = []
        loss_g_list = []

        self.init_weights()

        for epoch in range(self.n_epoches):
            torch.cuda.empty_cache()
            self.generator.train()
            self.discriminator.train()

            total_loss_g = 0.0
            total_loss_d = 0.0
            step = 0

            for m, y_norm in self.data_loader:
                m = m.to(self.device)
                y_norm = y_norm.to(self.device)

                u = self._uniform_sampler(m.shape)
                y_imp = m * y_norm + (1 - m) * u

                pred_y_norm = self.generator(y_imp, m)
                y0_com = m * y_norm + (1.0 - m) * pred_y_norm

                b = self._binary_sampler(m.shape, self.hint_rate)
                h = b * m + (1.0 - b) * 0.5

                #  Train Discriminator
                self.optim_d.zero_grad()
                prob_d = self.discriminator(y0_com.detach(), h)
                loss_d = self._discr_loss(prob_d, m)
                total_loss_d += loss_d.item()
                loss_d.backward()
                self.optim_d.step()

                ##  Train Generator
                self.optim_g.zero_grad()
                prob_d = self.discriminator(y0_com, h)
                pred_loss = self._prediction_loss(pred_y_norm, y_norm, m)
                fool_loss = self._fool_loss(prob_d, m)
                loss_g = self.coef * pred_loss + fool_loss
                total_loss_g += loss_g.item()
                loss_g.backward()
                self.optim_g.step()
                step += 1


            loss_d_list.append(total_loss_d)
            loss_g_list.append(total_loss_g)

            if (epoch + 1) % 10 == 0:
                print(f'epoch:{epoch + 1}, total loss for discriminator:{total_loss_d}, total loss for generator:{total_loss_g}')


        ## plot loss
        plt.subplot(2, 1, 1)
        x = range(len(loss_d_list))
        plt.plot(x, loss_d_list)
        plt.title('Discriminator loss vs. epoches')
        plt.ylabel('Discriminator loss')
        plt.subplot(2, 1, 2)
        plt.plot(x, loss_g_list)
        plt.title('Generator loss vs. epoches')
        plt.ylabel('Generator loss')
        ax = plt.gca()
        ax.ticklabel_format(style='plain')
        plt.show()


    def predict(self):
        self.generator.eval()
        res = []

        for m, y_norm in self.data_loader:
            m = m.to(self.device)
            y_norm = y_norm.to(self.device)
            b, T = y_norm.shape
            t = torch.arange(T).unsqueeze(0).repeat(b, 1).to(self.device)
            u = self._uniform_sampler(m.shape)
            y_imp = m * y_norm + (1 - m) * u
            pred_y0_norm = self.generator(y_imp, m)
            y_renorm = renormalization(y_norm, self.norm_parameters)
            pred_y0 = renormalization(pred_y0_norm, self.norm_parameters)
            res.append(torch.cat([t.unsqueeze(-1), y_norm.unsqueeze(-1), y_renorm.unsqueeze(-1), pred_y0.unsqueeze(-1)], dim=-1))

        res = torch.cat(res, dim=0)
        N, T, _ = res.shape
        res = torch.cat([torch.arange(N, device=self.device)[:,None,None].repeat(1, T, 1), res], dim=-1).reshape(N*T, -1)
        df = pd.DataFrame(res.cpu().detach().numpy(), columns=['unit', 'time', 'y_norm', 'y_renorm', 'pred_y0'])

        return df