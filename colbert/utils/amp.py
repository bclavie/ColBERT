import torch

from colbert.utils.utils import NullContextManager


class MixedPrecisionManager():
    def __init__(self, activated):
        self.activated = activated

        if self.activated:
            self.scaler = torch.amp.GradScaler("cuda")

    def context(self):
        return torch.amp.autocast("cuda") if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, colbert, optimizer, scheduler=None):
        if self.activated:
            self.scaler.unscale_(optimizer)
            if scheduler is not None:
                torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0, error_if_nonfinite=False)

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if scheduler is not None:
                torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
