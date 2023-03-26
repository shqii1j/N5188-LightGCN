import pdb

import world
import utils
from evaluate import test
from helper import early_stopping
import torch
import numpy as np
import random
from prettytable import PrettyTable
import time
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import wandb
from model_new import MixGCF

def get_feed_dict(train_entity_pairs, train_pos_set, n_items, start, end, n_negs=1, K=world.config["K"]):

    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs*K)).to(world.device)
    return feed_dict




wandb.init(sync_tensorboard=False,
               project="LightGCN",
               job_type="CleanRepo",
               config=world.config,
               )

lgn_model = register.MODELS[world.model_name]
model = MixGCF(world.config, dataset, lgn_model)
model = model.to(world.device)
bpr = utils.BPRLoss(model, world.config)

weight_file = "mixgcf_" + utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

train_cf = torch.LongTensor(np.dstack([dataset.trainUser, dataset.trainItem])[0])
train_cf_size = len(train_cf)


train_user_set = dict()
for u_id, i_id in zip(dataset.trainUser, dataset.trainItem):
    if train_user_set.get(u_id):
        train_user_set[u_id].append(i_id)
    else:
        train_user_set[u_id] = [i_id]

test_user_set = dataset.testDict
user_dict = {'train_user_set': train_user_set,
             'test_user_set': test_user_set}




optimizer = torch.optim.Adam(model.parameters(), lr=world.config['lr'])

cur_best_pre_0 = 0
stopping_step = 0
should_stop = False
print("loading model...")
# model.load_state_dict(torch.load(args.in_dir + 'model__' + '.ckpt'))
print("start training ...")
for epoch in range(world.TRAIN_epochs):
    # shuffle training data
    train_cf_ = train_cf
    index = np.arange(len(train_cf_))
    np.random.shuffle(index)
    train_cf_ = train_cf_[index].to(world.device)

    """training"""

    model.train()

    loss, s = 0, 0
    hits = 0
    train_s_t = time.time()
    while s + world.config['batch_size'] <= len(train_cf):
        batch = get_feed_dict(train_cf_,
                              user_dict['train_user_set'],
                              dataset.m_items,
                              s, s + world.config['batch_size'],
                              world.config['n_negs'])

        batch_loss, _, _ = model(batch)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss
        s += world.config['batch_size']

    train_e_t = time()
    wandb.log({"Loss": loss}, step=epoch + 1)

    """testing"""
    if epoch % 10 == 0:
        train_res = PrettyTable()
        train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg",
                                 "precision", "hit_ratio"]

        model.eval()
        test_s_t = time()
        test_ret = test(model, user_dict, mode='test')
        test_e_t = time()
        train_res.add_row(
            [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret['ndcg'],
             test_ret['precision'], test_ret['hit_ratio']])
        for i in range(len(world.topks)):
            wandb.log({str(world.topks[i]) + "/Precision": test_ret['precision'][i]}, step=epoch + 1)
            wandb.log({str(world.topks[i]) + "/Recall": test_ret["recall"][i]}, step=epoch + 1)
            wandb.log({str(world.topks[i]) + "/NDCG": test_ret["ndcg"][i]}, step=epoch + 1)
            wandb.log({str(world.topks[i]) + "/HitRate": test_ret['hit_ratio'][i]}, step=epoch + 1)



        print(train_res)

    # *********************************************************
    # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
    cur_best_pre_0, stopping_step, should_stop = early_stopping(test_ret['recall'][0], cur_best_pre_0,
                                                                stopping_step, expected_order='acc',
                                                                flag_step=10)
    if should_stop:
        break
wandb.finish()


