import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
torch.set_default_dtype(torch.float64)
# from torch.optim.scheduler

data = pd.read_csv('data/twins/twins_30_n1.csv')
X = torch.tensor(data[['x1', 'x2']].values)
Y = torch.tensor(data['sqJ'].values)

lr = 0.005
n_epochs = 50000
width = 64

model = torch.nn.Sequential(
    torch.nn.Linear(2,width),
    torch.nn.Tanh(),
    torch.nn.Linear(width,width),
    torch.nn.Tanh(),
    torch.nn.Linear(width,1),
)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


writer = SummaryWriter()
for e in range(n_epochs):
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = model(X).squeeze().sub(Y).abs().mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    if e%10 == 0:
        with torch.no_grad():
            loss = model(X).squeeze().sub(Y).abs().mean()
        writer.add_scalar('loss', loss,e)

