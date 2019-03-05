from utils import *
import logging
from aux_utils import compute_params
import re


def test_mode():
    args = get_arguments()
    args.enc_pretrained = True
    logger = logging.getLogger(__name__)
    torch.backends.cudnn.deterministic = True
    segmenter = create_segmenter(args.enc, args.enc_pretrained, args.num_classes[0]).cuda()
    segmenter.eval()
    logger.info(" Loaded Segmenter {}, ImageNet-Pre-Trained={}, #PARAMS={:3.2f}M"
                .format(args.enc, args.enc_pretrained, compute_params(segmenter) / 1e6))
    logger.info("Now you can do some test")
    while True:
        img_path = input("请输入文件名：")  # 'D:/VOCdata/VOC2012AUG/JPEGImages/2007_000027.jpg'
        if img_path == 'exit':
            break
        result = test_img(segmenter, img_path, args.num_classes[0], args.normalise_params)
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_mode()
