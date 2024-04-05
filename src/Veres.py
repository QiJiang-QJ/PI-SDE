from config_Veres import config
import train

args = config()

args.sigma_const = sigma
config = train.run(args)
