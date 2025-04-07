import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd


class Trainer():
    def __init__(self,
                 data_loader,
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

    def _fool_loss(self, prob_d, d):
        return -torch.mean(d * torch.log(prob_d + 1e-8))

    def _discr_loss(self, prob_d, d):
        return -torch.mean(d * torch.log(1.0 - prob_d + 1e-8) + (1.0 - d) * torch.log(prob_d + 1e-8))

    def _prediction_loss(self, pred_y0, real_y, d):
        diff = pred_y0 - real_y
        return torch.sum(torch.square(diff) * (1.0 - d)) / torch.sum(1.0 - d)

    def init_weights(self):
        for layer in self.generator.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

        for layer in self.discriminator.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)


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

            for d, y, y_norm, group in self.data_loader:
                d = d.to(self.device)
                y = y.to(self.device)
                y_norm = y_norm.to(self.device)
                group = group.to(self.device)

                y_imp = (1.0 - d) * y_norm + d * torch.zeros_like(y_norm)

                pred_y0 = self.generator(y_imp, group)
                y0_com = (1.0 - d) * y + d * pred_y0

                b = self._binary_sampler(d.shape, self.hint_rate)
                h = b * (1.0 - d) + (1.0 - b) * 0.5

                #  Train Discriminator
                self.optim_d.zero_grad()
                prob_d = self.discriminator(y0_com.detach(), h)
                loss_d = self._discr_loss(prob_d, d)
                total_loss_d += loss_d.item()
                loss_d.backward()
                self.optim_d.step()

                ##  Train Generator
                self.optim_g.zero_grad()
                prob_d = self.discriminator(y0_com, h)
                pred_loss = self._prediction_loss(pred_y0, y, d)
                fool_loss = self._fool_loss(prob_d, d)
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
        # plt.subplot(2, 1, 1)
        # x = range(len(loss_d_list))
        # plt.plot(x, loss_d_list)
        # plt.title('Discriminator loss vs. epoches')
        # plt.ylabel('Discriminator loss')
        # plt.subplot(2, 1, 2)
        # plt.plot(x, loss_g_list)
        # plt.title('Generator loss vs. epoches')
        # plt.ylabel('Generator loss')
        # ax = plt.gca()
        # ax.ticklabel_format(style='plain')
        # plt.show()


    def predict(self):
        self.generator.eval()
        res = []

        for d, y, y_norm, group in self.data_loader:
            d = d.to(self.device)
            y = y.to(self.device)
            y_norm = y_norm.to(self.device)
            group = group.to(self.device)
            y_imp = (1.0 - d) * y_norm + d * torch.zeros_like(y_norm)
            b, T = y.shape
            t = torch.arange(T).unsqueeze(0).repeat(b, 1).to(self.device)
            pred_y0 = self.generator(y_imp, group)
            res.append(torch.cat([t.unsqueeze(-1), y.unsqueeze(-1), pred_y0.unsqueeze(-1)], dim=-1))

        res = torch.cat(res, dim=0)
        N, T, _ = res.shape
        res = torch.cat([torch.arange(N, device=self.device)[:,None,None].repeat(1, T, 1), res], dim=-1).reshape(N*T, -1)
        df = pd.DataFrame(res.cpu().detach().numpy(), columns=['unit', 'time', 'y', 'pred_y0'])
        return df
