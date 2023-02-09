import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import wandb

wandb.init(sync_tensorboard=False,
               project="LightGCN",
               job_type="CleanRepo",
               config=world.config,
               )

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            test_result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            for i in range(len(world.topks)):
                wandb.log({str(world.topks[i])+"/Precision": test_result['precision'][i].detach().cpu()}, step=epoch)
                wandb.log({str(world.topks[i])+"/Recall": test_result["recall"][i].detach().cpu()}, step=epoch)
                wandb.log({str(world.topks[i])+"/NDCG": test_result["ndcg"][i].detach().cpu()}, step=epoch)

        aver_loss, time_info = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] loss{aver_loss:.3f}-{time_info}')
        wandb.log({"Loss": aver_loss.detach().cpu()})
        torch.save(Recmodel.state_dict(), weight_file)

finally:
    if world.tensorboard:
        w.close()
