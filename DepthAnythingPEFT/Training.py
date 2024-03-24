import Dataset
from Evaluation import EvaluationMetric
import torch
from torch.utils.data import Dataset
import wandb
import math
import os


class PEFTTraining:

    def __init__(self, model_checkpoint: str,output_dir: str, model, train_dataset: Dataset, 
                 valid_dataset: Dataset, train_batch_size:int, valid_batch_size:int , 
                 loss_fn, optimizer, epoch: int, device, warmup_period_percentage,
                 learning_rate,min_lr,grad_clip, wandb_logging = True) -> None:

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model = model
        self.epoch = epoch
        self.output_dir = output_dir
        self.logging = wandb_logging
        self.eval = EvaluationMetric(wandb_logging)
        self.device = device
        self.model.to(self.device)
        self.train_batch_size = train_batch_size
        self.model_name = model_checkpoint.split("/")[-1]
        self.training_loader = torch.utils.data.DataLoader(train_dataset, batch_size= train_batch_size, shuffle=True)
        self.validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size= valid_batch_size, shuffle=False)
        self.num_steps = len(self.training_loader) * self.epoch
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.grad_clip = grad_clip
        self.iter_per_epoch = len(train_dataset)/train_batch_size
        self.lr_decay_iters = int(self.iter_per_epoch * epoch) + 1
        self.warmup_period = self.lr_decay_iters*warmup_period_percentage/100
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=T_mult)
        # self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=self.warmup_period)
        
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self,iteration):
        # 1) linear warmup for warmup_iters steps
        if iteration < self.warmup_period:
            return self.learning_rate * iteration / self.warmup_period
        # 2) if it > lr_decay_iters, return min learning rate
        if iteration > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iteration - self.warmup_period) / (self.lr_decay_iters - self.warmup_period)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)


    def train_one_epoch(self, epoch_index):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        iteration = epoch_index * self.iter_per_epoch
        for i, (inputs, labels) in enumerate(self.training_loader):
            # Every data instance is an input + label pair
            inputs, labels = inputs.to(self.device),labels.to(self.device)

            # determine and set the learning rate for this iteration
            lr = self.get_lr(iteration)
            self.optimizer.param_groups[0]["lr"] = lr

            # clip the gradient
            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs).predicted_depth

            # Compute the loss and its gradients
            
            loss = self.loss_fn(torch.squeeze(outputs), torch.squeeze(labels))
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # with self.warmup_scheduler.dampening():
            #     if self.warmup_scheduler.last_step + 1 >= self.warmup_period:
            #         self.lr_scheduler.step()

            # Gather data and report
          
            running_loss += loss.item()
            iteration += 1
            if i % self.train_batch_size == self.train_batch_size - 1:
                last_loss = running_loss / self.train_batch_size
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                #wb_x = epoch_index * len(self.training_loader) + i + 1
                wandb.log({'Loss/train (per batch)': last_loss})
                wandb.log({'Learning Rate (per batch)':  self.optimizer.param_groups[0]["lr"]})
                running_loss = 0.

        return last_loss
    


    def train(self,wandb_init = None):

        epoch_number = 0

        best_vloss = 1_000_000.

        for epoch in range(self.epoch):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number)


            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, (vinputs, vlabels) in enumerate(self.validation_loader):
                    vinputs, vlabels = vinputs.to(self.device),vlabels.to(self.device)
                    voutputs = self.model(vinputs).predicted_depth
                    vloss = self.loss_fn(torch.squeeze(voutputs), torch.squeeze(vlabels))
                    self.eval.compute_metrics(vinputs,voutputs,vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            if self.logging == True:
                wandb.log({ 'Training Loss' : avg_loss, 'Validation Loss' : avg_vloss })

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)

                model_path = '{}/{}_{}.pth'.format(self.output_dir, self.model_name, epoch_number)
                torch.save({'epoch': epoch_number,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': best_vloss}, 
                            model_path)
                
                if self.logging == True:
                    if wandb_init is not None:
                        artifact = wandb.Artifact('model', type='model')
                        artifact.add_file(model_path)
                        wandb_init.log_artifact(artifact)
                    else:
                        print("No WandB init given; model artifact is not saved")


            epoch_number += 1



