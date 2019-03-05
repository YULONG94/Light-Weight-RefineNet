import torch
import json
import os


def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os, sys
    from six.moves import urllib
    if model_dir is None:
        # torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = './models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)



def compute_params(model):
    """Compute number of parameters"""
    n_total_params = 0
    for name, m in model.named_parameters():
        n_elem = m.numel()
        n_total_params += n_elem
    return n_total_params


# Adopted from https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

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


class Saver():
    """Saver class for managing parameters"""

    def __init__(self, args, ckpt_dir, best_val=0, condition=lambda x, y: x > y):
        """
        Args:
            args (dict): dictionary with arguments.
            ckpt_dir (str): path to directory in which to store the checkpoint.
            best_val (float): initial best value.
            condition (function): how to decide whether to save the new checkpoint
                                    by comparing best value and new value (x,y).

        """
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        with open('{}/args.json'.format(ckpt_dir), 'w') as f:
            json.dump({k: v for k, v in args.items() if isinstance(v, (int, float, str))}, f,
                      sort_keys=True, indent=4, ensure_ascii=False)
        self.ckpt_dir = ckpt_dir
        self.best_val = best_val
        self.condition = condition
        self._counter = 0

    def _do_save(self, new_val):
        """Check whether need to save"""
        return self.condition(new_val, self.best_val)

    def save(self, new_val, dict_to_save, logger):
        """Save new checkpoint"""
        self._counter += 1
        if self._do_save(new_val):
            logger.info(" New best value {:.4f}, was {:.4f}".format(new_val, self.best_val))
            self.best_val = new_val
            dict_to_save['best_val'] = new_val
            torch.save(dict_to_save, '{}/checkpoint.pth.tar'.format(self.ckpt_dir))
            return True
        return False
