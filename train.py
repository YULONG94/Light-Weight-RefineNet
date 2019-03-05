from utils import *
import logging
import random
from aux_utils import Saver, compute_params
import re


def main():
    args = get_arguments()
    logger = logging.getLogger(__name__)
    ## Add args ##
    args.num_stages = len(args.num_classes)
    ## Set random seeds ##
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    ## Generate Segmenter ##
    segmenter = create_segmenter(args.enc, args.enc_pretrained, args.num_classes[0]).cuda()
    logger.info(" Loaded Segmenter {}, ImageNet-Pre-Trained={}, #PARAMS={:3.2f}M"
                .format(args.enc, args.enc_pretrained, compute_params(segmenter) / 1e6))
    ## Restore if any ##
    best_val, epoch_start = load_ckpt(args.ckpt_path, {'segmenter': segmenter})
    ## Criterion ##
    segm_crit = nn.NLLLoss2d(ignore_index=args.ignore_label).cuda()

    ## Saver ##
    saver = Saver(args=vars(args),
                  ckpt_dir=args.snapshot_dir,
                  best_val=best_val,
                  condition=lambda x, y: x > y)  # keep checkpoint with the best validation score

    logger.info(" Training Process Starts")
    for task_idx in range(args.num_stages):
        start = time.time()
        torch.cuda.empty_cache()
        ## Create dataloaders ##
        train_loader, val_loader = create_loaders(args.train_dir,
                                                  args.val_dir,
                                                  args.train_list[task_idx],
                                                  args.val_list[task_idx],
                                                  args.shorter_side[task_idx],
                                                  args.crop_size[task_idx],
                                                  args.low_scale[task_idx],
                                                  args.high_scale[task_idx],
                                                  args.normalise_params,
                                                  args.batch_size[task_idx],
                                                  args.num_workers,
                                                  args.ignore_label)
        if args.evaluate:
            return validate(segmenter, val_loader, 0, num_classes=args.num_classes[task_idx])
        logger.info(" Training Stage {}".format(str(task_idx)))
        ## Optimisers ##
        enc_params = []
        dec_params = []
        for k, v in segmenter.named_parameters():
            if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
                enc_params.append(v)
                logger.info(" Enc. parameter: {}".format(k))
            else:
                dec_params.append(v)
                logger.info(" Dec. parameter: {}".format(k))
        optim_enc, optim_dec = create_optimisers(args.lr_enc[task_idx], args.lr_dec[task_idx],
                                                 args.mom_enc[task_idx], args.mom_dec[task_idx],
                                                 args.wd_enc[task_idx], args.wd_dec[task_idx],
                                                 enc_params, dec_params, args.optim_dec)
        for epoch in range(args.num_segm_epochs[task_idx]):
            train_segmenter(segmenter, train_loader,
                            optim_enc, optim_dec,
                            epoch_start, segm_crit,
                            args.freeze_bn[task_idx], args.print_every)
            if (epoch + 1) % 100 == 0:
                torch.save(segmenter, args.model_path)
            if (epoch + 1) % (args.val_every[task_idx]) == 0:
                miou = validate(segmenter, val_loader, epoch_start, args.num_classes[task_idx])
                saver.save(
                    miou,
                    {'segmenter': segmenter.state_dict(),
                     'epoch_start': epoch_start}, logger
                )
            epoch_start += 1
        logger.info("Stage {} finished, time spent {:.3f}min".format(
            task_idx, (time.time() - start) / 60.))
    logger.info("All stages are now finished. Best Val is {:.3f}".format(
        saver.best_val))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
