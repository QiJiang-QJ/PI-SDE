from src.config_Veres import config, init_config
import src.train as train
from src.evaluation import evaluate_fit_leaveout

args = config()

args.task = 'leaveout'
args.data_path = 'data/Veres/leaveout7/fate_train.pt'


config = train.run_leaveout(args, init_config, leaveouts=[7])
evaluate_fit_leaveout(config, init_config, leaveouts=[7], use_loss='emd')


# config = train.run_leaveout(args, init_config,leaveouts=[3,4])
# evaluate_fit_leaveout(config, init_config,leaveouts=[3,4],use_loss='emd')

