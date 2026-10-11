import importlib
from easydict import EasyDict
from .__init__ import pretrained_models_path


class ParamManager:
    
    def __init__(self, args):  # 构造函数，传入参数 args

        # 为 args 对象添加 text_pretrained_model 属性，其值为从 pretrained_models_path 中获取的对应文本骨干网络的预训练模型路径
        # args.text_backbone 应该是指定的文本骨干网络名称，用于索引 pretrained_models_path 字典获取对应的预训练模型路径
        args.text_pretrained_model = pretrained_models_path[args.text_backbone]

        # 将 args 对象转换为字典，再用 EasyDict 包装，方便后续像访问属性一样访问参数
        self.args = EasyDict(dict(vars(args)))   
        
def add_config_param(old_args, config_file_name = None):
    # 添加配置文件中的参数到旧参数中

    if config_file_name is None:
        config_file_name = old_args.config_file_name
        
    if config_file_name.endswith('.py'):
        module_name = '.' + config_file_name[:-3]
    else:
        module_name = '.' + config_file_name

    config = importlib.import_module(module_name, 'configs')

    config_param = config.Param
    method_args = config_param(old_args)
    new_args = EasyDict(dict(old_args, **method_args.hyper_param))
    
    return new_args
