from configs.base import ParamManager, add_config_param
from data.base import DataManager
from methods import method_map
from backbones.base import ModelManager
from utils.functions import set_torch_seed, save_results, set_output_path

import argparse
import logging
import os
import datetime
import itertools
import warnings
import copy


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logger_name', type=str, default='umc-icla',
                        help="Logger name for multimodal semantics discovery.")

    parser.add_argument('--dataset', type=str, default='MIntRec', help="The selected dataset.")

    parser.add_argument('--multimodal_method', type=str, default='umc-icla', help="which method to use.")

    parser.add_argument('--method', type=str, default='umc-icla', help="which method to use.")

    parser.add_argument("--text_backbone", type=str, default='bert-base-uncased',
                        help="which backbone to use for text modality")

    # parser.add_argument('--seed', type=int, default=0, nargs='+', help="The random seeds to use.")
    parser.add_argument('--seed', type=int, default=0, help="The random seeds to use.")

    parser.add_argument('--num_workers', type=int, default=4, help="The number of workers to load data.")

    parser.add_argument('--log_id', type=str, default=None, help="The index of each logging file.")

    parser.add_argument('--gpu_id', type=str, default='0', help="The selected person id.")

    parser.add_argument("--data_path", default='Datasets', type=str,
                        help="The input data dir. Should contain text, video and audio data for the task.")

    parser.add_argument("--train", action="store_true", default=True, help="Whether to train the model.")

    parser.add_argument("--tune", action="store_true", default=True,
                        help="Whether to tune the model with a series of hyper-parameters.")

    parser.add_argument("--save_model", action="store_true", default=True,
                        help="save trained-model for multimodal intent recognition.")

    parser.add_argument("--save_results", action="store_true", default=True,
                        help="save final results for multimodal intent recognition.")

    parser.add_argument('--log_path', type=str, default='logs', help="Logger directory.")

    parser.add_argument('--cache_path', type=str, default='cache', help="The caching directory for pre-trained models.")

    parser.add_argument('--video_data_path', type=str, default='video_data', help="The directory of the video data.")

    parser.add_argument('--audio_data_path', type=str, default='audio_data', help="The directory of the audio data.")

    parser.add_argument('--video_feats_path', type=str, default='swin_feats.pkl',
                        help="The directory of the video features.")

    parser.add_argument('--audio_feats_path', type=str, default='wavlm_feats.pkl',
                        help="The directory of the audio features.")

    parser.add_argument('--results_path', type=str, default='results', help="The path to save results.")

    parser.add_argument("--output_path", default="outputs/MIntRec", type=str,
                        help="The output directory where all train data will be written.")

    parser.add_argument("--model_path", default='models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--config_file_name", type=str, default='umc_icla_MIntRec.py', help="The name of the config file.")

    parser.add_argument("--results_file_name", type=str, default="results_umc_icla.csv",
                        help="The file name of all the results.")

    args = parser.parse_args()

    return args


def set_logger(args):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.logger_name = f"{args.method}_{args.multimodal_method}_{args.text_backbone}_{args.dataset}_{args.seed}"
    args.log_id = f"{args.logger_name}_{time}"

    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(args.log_path, args.log_id + '.log')
    fh = logging.FileHandler(log_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def set_up(args):
    save_model_name = f"{args.method}_{args.multimodal_method}_{args.dataset}_{args.text_backbone}_{args.seed}"

    args.pred_output_path, args.model_output_path = set_output_path(args, save_model_name)

    set_torch_seed(args.seed)

    return args


def work(args, data, logger, debug_args=None):
    # 设置随机种子以确保结果可重现或随机性控制
    set_torch_seed(args.seed)

    # 根据 args.method 从 method_map 中获取对应的方法管理器类
    # 例如，--method umc-icla 会从 .unsupervised.UMC_ICLA.manager 调取管理器。
    method_manager = method_map[args.method]
    model = ModelManager(args)  # 初始化模型管理器
    method = method_manager(args, data, model)  # 使用方法管理器、数据和模型创建方法对象

    logger.info('Multimodal Intent Recognition begins...')  # 记录日志：多模态意图识别开始

    # 前面数据预训练结束后，进行模型训练与测试
    if args.train:
        logger.info('Training begins...')  # 如果 args.train 为真，则开始训练过程
        method._train(args)
        logger.info('Training is finished...')  # 记录日志：训练结束

    logger.info('Testing begins...')  # 开始测试过程
    outputs = method._test(args)
    logger.info('Testing is finished...')  # 记录日志：测试结束
    logger.info('Multimodal intent recognition is finished...')  # 记录日志：多模态意图识别完成

    # 如果需要保存结果，则调用 save_results 函数保存结果
    if args.save_results:
        logger.info('Results are saved in %s', str(os.path.join(args.results_path, args.results_file_name)))
        save_results(args, outputs, debug_args=debug_args)


def run(args, data, logger, ind_args=None):
    # seeds = [0, 1]  # 在这里设置种子列表的值

    debug_args = {}

    # 收集 args 中所有值为列表的键值对
    for k, v in args.items():
        if isinstance(v, list):
            debug_args[k] = v

    # 生成调试参数的所有可能组合
    for result in itertools.product(*debug_args.values()):
        # 遍历每个参数组合
        for i, key in enumerate(debug_args.keys()):
            # 将参数组合中的值分配回 args 对应的键
            args[key] = result[i]

        # 遍历每个种子值
        # for seed in seeds:
        #     args['seed'] = seed
        #     work(args, data, logger, debug_args)
        work(args, data, logger, debug_args)


if __name__ == '__main__':

    warnings.filterwarnings('ignore')  # 忽略所有的警告信息

    args = parse_arguments()  # 调用 parse_arguments 函数来解析命令行参数，args 将包含用户在运行脚本时提供的各种参数
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # 设置环境变量 CUDA_VISIBLE_DEVICES，启用GPU

    param = ParamManager(args)  # 创建 ParamManager 对象，传入解析后的参数 args，ParamManager 对象是参数管理器类，用于处理和管理参数
    args = param.args  # 将处理后的参数重新赋值给 args

    data = DataManager(args)  # 创建 DataManager 对象，传入参数 args，，，DataManager 可能负责数据的加载、预处理等工作
    logger = set_logger(args)  # 调用 set_logger 函数来设置日志记录器，日志记录器用于输出程序运行过程中的各种信息

    args = add_config_param(args, args.config_file_name)  # 调用 add_config_param 函数向 args 添加配置文件中的参数，这允许程序从配置文件（args.config_file_name）中读取额外的参数设置
    args = set_up(args)  # 调用 set_up 函数对 args 进行进一步的设置或处理

    logger.info("=" * 30 + " Params " + "=" * 30)  # 使用日志记录器输出一道分隔符，方便查看参数信息
    for k in args.keys():
        logger.info(f"{k}: {args[k]}")  # 遍历 args 中的所有键值对，使用日志记录器输出每个参数的名称和对应的值
    logger.info("=" * 30 + " End Params " + "=" * 30)  # 输出另一道分隔符，表示参数信息输出结束

    # 正式开始
    # 正式开始
    # 正式开始
    run(args, data, logger)
