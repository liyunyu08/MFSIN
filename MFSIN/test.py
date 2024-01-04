import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.msfin import MSFIN
import numpy as np


def evaluate(epoch, model, loader, args=None, set='val'):
    model.eval()

    loss_meter = Meter()
    acc_meter = Meter()

    label = torch.arange(args.way).repeat(args.query).cuda()

    k = args.way * args.shot
    q = args.way * args.query
    tqdm_gen = tqdm.tqdm(loader)

    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm_gen, 1):
            data = data.cuda()

            data, train_labels = data.cuda(), label.cuda()

            model.module.mode = 'encoder'
            data1,data2 = model(data)
            data_shot1, data_query1 = data1[:k], data1[k:]
            data_shot2, data_query2 = data2[:k], data2[k:]

            if args.shot > 1:
                data_shot1 = data_shot1.view(args.shot, args.way, *data_shot1.shape[1:])
                data_shot2 = data_shot2.view(args.shot, args.way, *data_shot2.shape[1:])
                data_shot1 = data_shot1.mean(dim=0)
                data_shot2 = data_shot2.mean(dim=0)

            model.module.mode = 'local_feat1'
            local_shot1 = model(data_shot1)
            local_query1 = model(data_query1)

            model.module.mode = 'local_feat2'
            local_shot2 = model(data_shot2)
            local_query2 = model(data_query2)

            model.module.mode = 'msfn1'
            glo_shot1 = model(data_shot1)
            glo_query1 = model(data_query1)
            sele_local_shot1 = model(local_shot1)
            sele_local_query1 = model(local_query1)
            sele_local_shot1, sele_local_query1 = local_shape(sele_local_shot1, sele_local_query1,args)

            model.module.mode = 'msfn2'
            glo_shot2 = model(data_shot2)
            glo_query2 = model(data_query2)
            sele_local_shot2 = model(local_shot2)
            sele_local_query2 = model(local_query2)
            sele_local_shot2, sele_local_query2 = local_shape(sele_local_shot2, sele_local_query2,args)

            all_shot = torch.cat((glo_shot1, glo_shot2, sele_local_shot1, sele_local_shot2), dim=-1)
            all_qry = torch.cat((glo_query1, glo_query2, sele_local_query1, sele_local_query2), dim=-1)

            model.module.mode = 'mfin'
            logits = model((all_shot, all_qry))
            loss = F.cross_entropy(logits, label)
            acc = compute_accuracy(logits, label)

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(
                f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def local_shape(sele_local_shot,sele_local_query,args):
    _, c, n = sele_local_shot.shape
    sele_local_shot = sele_local_shot.reshape(args.way, -1, c, n)
    sele_local_shot = sele_local_shot.permute(0, 2, 1, 3).reshape(args.way, c, -1)

    sele_local_query = sele_local_query.reshape(args.way * args.query, -1, c, n)
    sele_local_query = sele_local_query.permute(0, 2, 1, 3).reshape(args.way * args.query, c, -1)

    return sele_local_shot,sele_local_query


def test_main(model, args):
    ''' load model '''
    model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))

    ''' define test dataset '''
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    ''' evaluate the model with the dataset '''
    _, test_acc, test_ci = evaluate("best", model, test_loader, args, set='test')
    print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')

    return test_acc, test_ci


if __name__ == '__main__':
    args = setup_run(arg_mode='test')

    ''' define model '''
    model = MSFIN(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)
    test_main(model, args)
