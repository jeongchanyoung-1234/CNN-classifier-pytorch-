import time
import pprint

import torch


class FashionMNISTTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 train_dataloader,
                 valid_dataloader,
                 config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.config = config

        self.best_epoch = 0
        self.best_loss = float('inf')
        self.best_acc = 0

    def print_config(self) -> None:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint('Model',
                  self.model,
                  'Optimizer',
                  self.optimizer,
                  'Criterion',
                  self.criterion)

    def train(self):
        total_train_loss, total_valid_loss, total_train_acc, total_valid_acc, train_cnt, valid_cnt = 0, 0, 0, 0, 0, 0

        start = time.time()
        # train loop
        for epoch in range(self.config.epochs):
            self.model.train()
            for i, (batch_x, batch_y) in enumerate(self.train_dataloader):
                y_hat = self.model(batch_x)
                loss = self.criterion(y_hat, batch_y)
                acc = (y_hat.argmax(-1) == batch_y).sum().item() / len(batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss
                total_train_acc += acc
                train_cnt += 1

            avg_train_loss = total_train_loss / train_cnt
            avg_train_acc = total_train_acc / train_cnt

            self.model.eval()
            for i, (batch_x, batch_y) in enumerate(self.valid_dataloader):
                with torch.no_grad():
                    y_hat = self.model(batch_x)
                    loss = self.criterion(y_hat, batch_y)
                    acc = (y_hat.argmax(-1) == batch_y).sum().item() / len(batch_y)

                    total_valid_loss += loss
                    total_valid_acc += acc
                    valid_cnt += 1

            avg_valid_loss = total_valid_loss / valid_cnt
            avg_valid_acc = total_valid_acc / valid_cnt

            if self.best_acc < avg_valid_acc:
                self.best_acc = avg_valid_acc
                self.best_loss = avg_valid_loss
                self.best_epoch = epoch + 1

            if (epoch + 1) % self.config.verbose == 0:
                print('|EPOCH ({}/{})| train_loss={:.4f} train_acc={:2.2f} valid_loss={:.4f} valid_acc={:2.2f}  ({:2.2f}sec)'.format(
                    epoch + 1,
                    self.config.epochs,
                    avg_train_loss,
                    avg_train_acc * 100,
                    avg_valid_loss,
                    avg_valid_acc * 100,
                    time.time() - start
                ))

            total_train_loss, total_valid_loss, total_train_acc, total_valid_acc, train_cnt, valid_cnt = 0, 0, 0, 0, 0, 0

        print('''
        |Training completed succesfully|
        Best epoch={}
        Best loss={}
        Best Accuracy={}
        '''.format(self.best_epoch, self.best_loss, self.best_acc))





