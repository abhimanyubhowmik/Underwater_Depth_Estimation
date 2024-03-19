import Dataset
from Evaluation import EvaluationMetric
import torch
from torch.utils.data import Dataset
import wandb



class PEFTTraining:

    def __init__(self, model_checkpoint: str,output_dir: str, model, train_dataset: Dataset, 
                 valid_dataset: Dataset, train_batch_size:int, valid_batch_size:int , 
                 loss_fn, optimizer,scheduler, epoch: int, device, wandb_logging = True) -> None:

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model = model
        self.epoch = epoch
        self.output_dir = output_dir
        self.logging = wandb_logging
        self.eval = EvaluationMetric(wandb_logging)
        self.device = device
        self.model.to(self.device)
        self.model_name = model_checkpoint.split("/")[-1]
        self.training_loader = torch.utils.data.DataLoader(train_dataset, batch_size= train_batch_size, shuffle=True)
        self.validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size= valid_batch_size, shuffle=False)
        self.lr_scheduler = scheduler
        


    def train_one_epoch(self, epoch_index):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, (inputs, labels) in enumerate(self.training_loader):
            # Every data instance is an input + label pair
            inputs, labels = inputs.to(self.device),labels.to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs).predicted_depth

            # Compute the loss and its gradients
            
            loss = self.loss_fn(torch.squeeze(outputs), torch.squeeze(labels))
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Adjust the learning rate 
            self.lr_scheduler.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 16 == 15:
                last_loss = running_loss / 16 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                wb_x = epoch_index * len(self.training_loader) + i + 1
                wandb.log({'Loss/train (per batch)': last_loss}, step = wb_x)
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
                wandb.log({ 'Training' : avg_loss, 'Validation' : avg_vloss })

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
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



