" Target Unit Head. "

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.lib import utils as L

mask_prior = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-7, -8.5, -7.5, -7.5, -7.5, -7.5, -4, -2.5, -4.5, -5, -2, -1, -3, -9, -10, -9, -8, -10, -6, -6.5, -6.5, -10],
                     [-7, -8.5, -7.5, -7.5, -7.5, -7.5, -4, -2.5, -4.5, -5, -2, -1, -3, -9, -10, -9, -8, -10, -6, -6.5, -6.5, -11],
                     [-7, -8.5, -7.5, -7.5, -7.5, -7.5, -4, -2.5, -4.5, -5, -2, -1, -3, -10, -9, -10, -8, -9, -6, -6.5, -6.5, -11],
                     [-7, -8.5, -7.5, -7.5, -7.5, -7.5, -4, -2.5, -4.5, -5, -2, -1, -3, -10, -9, -10, -8, -9, -6, -6.5, -6.5, -11],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -2, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -2, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -2, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -2, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -2, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -3, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -2, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -2, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -2, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -3, -4, -2, -10, -10, -10, -10, -10, -10, -10, -10, -10],
                     ]).cuda()

class TargetUnitHead(nn.Module):
    '''
    Inputs: autoregressive_embedding, action_type, entity_embeddings
    Outputs:
        target
    '''

    def __init__(self, ori_128=64, ori_32=32):
        super(TargetUnitHead, self).__init__()
        self.func_embed = nn.Linear(5, ori_128)
        self.conv_1 = nn.Conv1d(in_channels=ori_128, out_channels=ori_32, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc_1 = nn.Linear(ori_128, ori_128)
        self.fc_2 = nn.Linear(ori_128, ori_32)
        self.fc_3 = nn.Linear(5, ori_32)
        self.fc_4 = nn.Linear(ori_32, ori_128)
        self.fc_5 = nn.Linear(ori_128, ori_128)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, autoregressive_embedding, entity_embeddings, num, obs, task_now, task_context, task_num_now):
        entity_embeddings = entity_embeddings[:, -num:, :]
        key = self.conv_1(entity_embeddings.transpose(-1, -2)).transpose(-1, -2)

        x = self.fc_1(autoregressive_embedding)
        query = self.fc_2(x).unsqueeze(0)

        y = torch.bmm(query, key.transpose(1, 2))

        mask = self.process_mask_base_on_task(obs, task_now, task_context, task_num_now)
        target_unit_logits = y.squeeze(0) + mask
        target_unit_probs = self.softmax(target_unit_logits)
        target_unit_id = torch.multinomial(target_unit_probs, 1)

        return target_unit_logits, target_unit_probs, target_unit_id

    def process_mask_base_on_task(self, obs, task_now, task_context, task_num_now):
        mask = torch.ones([53, 22]).cuda() * -500
        for i in range(len(task_now)):
            if task_now[i] == 100 or task_now[i] == -1:
                pass
            elif task_context[i][task_num_now[i]][0] == 'attack':
                if task_context[i][task_num_now[i]][2] == 'infantry':
                    mask[i, 6:10] = 0
                elif task_context[i][task_num_now[i]][2] == 'artillery':
                    mask[i, 10:13] = 0
                elif task_context[i][task_num_now[i]][2] == 'all':
                    mask[i, :] = 0

        index = 0
        life = [1 for _ in range(22)]
        for id, entities in obs['blueSituation'].items():
            for entity in entities:
                if entity['life'] <= 0:
                    mask[:, index] = -1000
                    life[index] = 0
                index += 1

        mask = mask + mask_prior * 10

        return mask

    def process_mask(self, obs, mask):
        index = 0
        life = [1 for _ in range(22)]
        for id, entities in obs['blueSituation'].items():
            for entity in entities:
                if entity['life'] <= 0:
                    mask[:, index] = -1000
                    life[index] = 0
                index += 1
        att_other = [1, 2, 3, 4, 5, 13, 14, 15, 16, 19]
        att_list = [9, 18]
        att_list_ = [0, 19, 20]
        if life[0] == 0 and life[19] == 0 and life[20] == 0:
            mask[12:14, att_other] = 0
        elif life[9] == 0 and life[18] == 0:
            mask[12:14, att_list_] = 0
        elif life[7] == 0 and life[11] == 0:
            mask[12:14, att_list] = 0
        elif life[7] == 0:
            mask[12:14, 11] = 0

        att_list = [0, 19, 20]
        if life[0] == 0 and life[19] == 0 and life[20] == 0:
            mask[11, att_other] = 0
        if life[6] == 0 and life[10] == 0:
            mask[11, att_list] = 0
        elif life[6] == 0:
            mask[11, 10] = 0

        att_list = [0, 19, 20]
        if life[0] == 0 and life[19] == 0 and life[20] == 0:
            mask[14, att_other] = 0
        if life[8] == 0 and life[12] == 0:
            mask[14, att_list] = 0
        elif life[8] == 0:
            mask[14, 12] = 0

        index = 0
        life = [1 for _ in range(22)]
        for id, entities in obs['blueSituation'].items():
            for entity in entities:
                if entity['life'] <= 0:
                    mask[:, index] = -1000
                    life[index] = 0
                index += 1

        mask = mask + mask_prior
        return mask
