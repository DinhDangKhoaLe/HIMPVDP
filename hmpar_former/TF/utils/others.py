import torch
import torch.optim as optim

from models.lstm_vel_posetrack import LSTMVelPosetrack
from models.lstm_vel_3dpw import LSTMVel3dpw
from models.de_global_posetrack import DEGlobalPosetrack
from models.de_local_posetrack import DELocalPosetrack

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_model(opt,input_len,output_len):
    return LSTMVel3dpw(opt,input_len,output_len).to(opt.device)



def load_model(opt, input_len,output_len, load_ckpt=None):
    if load_ckpt:
        ckpt = torch.load(load_ckpt, map_location='cpu')
    else:
        ckpt = torch.load(opt.load_ckpt, map_location='cpu')

    ckpt_opt = ckpt['opt']
    for key, val in ckpt_opt.__dict__.items():
        setattr(opt, key, val)
    model = set_model(opt,input_len,output_len)
    state_dict = ckpt['model']
    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def set_optimizer(opt, model):
    return optim.Adam(model.parameters(), lr=opt.learning_rate)


def set_scheduler(opt, optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=opt.lr_decay_rate, patience=25, threshold=1e-8,
                                                verbose=True)

def speed2pos(preds, obs_p):
    pred_pos = torch.zeros(preds.shape[0], preds.shape[1], preds.shape[2]).to('cuda')
    current = obs_p[:, -1, :]

    for i in range(preds.shape[1]):
        pred_pos[:, i, :] = current + preds[:, i, :]
        current = pred_pos[:, i, :]

    for i in range(preds.shape[2]):
        pred_pos[:, :, i] = torch.min(pred_pos[:, :, i],
                                      1920 * torch.ones(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))
        pred_pos[:, :, i] = torch.max(pred_pos[:, :, i],
                                      torch.zeros(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))

    return pred_pos


def speed2pos_local(preds, obs_p):
    pred_pos = torch.zeros(preds.shape[0], preds.shape[1], preds.shape[2]).to('cuda')
    current = obs_p[:, -1, :]

    for i in range(preds.shape[1]):
        pred_pos[:, i, :] = current + preds[:, i, :]
        current = pred_pos[:, i, :]

    return pred_pos


def speed2pos3d(preds, obs_p):
    pred_pos = torch.zeros(preds.shape[0], preds.shape[1], preds.shape[2]).to('cuda')
    current = obs_p[:, -1, :]

    #Only Use Hand Positions for Future Pose
    # current = torch.stack([torch.cat((t[:3], t[-3:])) for t in current])
      
    for i in range(preds.shape[1]):
        pred_pos[:, i, :] = current + preds[:, i, :]
        current = pred_pos[:, i, :]

    return pred_pos
