from src.config_Veres import config, init_config
import src.train as train
from src.evaluation import evaluate_fit


args = config()
config = train.run(args, init_config)
evaluate_fit(config, init_config, use_loss='emd')
