from object_detection.utils import config_util

configs = config_util.get_configs_from_pipeline_file("pipeline.config")

model_config = configs['model']
train_config = configs['train_config']
input_config = configs['train_input_config']



print( dir(model_config))
#print(  model_config )
print( getattr(model_config, model_config.WhichOneof("model")) )
