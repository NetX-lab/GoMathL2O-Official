"""
File:    utils.py
Created: December 2, 2019
Revised: December 2, 2019
Authors: Howard Heaton, Xiaohan Chen
Purpose: Definition of different utility functions used in other files, e.g.,
         logging handler.
"""

import logging
import torch


def setup_logger(log_file):
    if log_file is not None:
        logging.basicConfig(filename=log_file, level=logging.INFO)
        lgr = logging.getLogger()
        lgr.addHandler(logging.StreamHandler())
        lgr = lgr.info
    else:
        lgr = print

    return lgr


def rand_WY(opts):
    def generate_WY(_batch_size, iterations, seed):
        Ws, Ys = [], []
        for i in range(iterations):
            # torch.manual_seed(seed + i)
            _W = torch.randn(_batch_size, opts.output_dim,
                             opts.input_dim, dtype=torch.float32)
            _W = _W / torch.sum(_W**2, dim=1, keepdim=True).sqrt()
            # torch.manual_seed(seed + i)
            _X_gt = torch.randn(_batch_size, opts.input_dim,
                                dtype=torch.float32)
            # torch.manual_seed(seed + i)
            non_zero_idx = torch.multinomial(
                torch.ones_like(_X_gt), num_samples=opts.sparsity, replacement=False
            )
            X_gt = torch.zeros_like(_X_gt).scatter(
                dim=1, index=non_zero_idx, src=_X_gt
            ).unsqueeze(-1)
            _Y = torch.bmm(_W, X_gt)
            Ws.append(_W)
            Ys.append(_Y)
        return torch.concat(Ws, dim=0), torch.concat(Ys, dim=0)

    W, Y = generate_WY(opts.train_batch_size,
                       opts.global_training_steps, opts.seed + 77)

    eval_W, eval_Y = generate_WY(opts.val_size, 1, opts.seed + 650)
    return W, Y, eval_W, eval_Y
