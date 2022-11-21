from pathlib import Path

class Default(object):
    
    n_envs = 64
    lr_value = 0.001 
    lr_policy = 0.001
    gamma = 0.99
    beta = 0.01
    load_path = None
    save_path = None
    save = False

    
    
def get_config(FLAGS):
    
    config = Default()
    
    config.n_envs = config.n_envs if FLAGS.n_envs is None else FLAGS.n_envs
    config.lr_value = config.lr_value if FLAGS.lr_value is None else FLAGS.lr_value
    config.lr_policy = config.lr_policy if FLAGS.lr_policy is None else FLAGS.lr_policy
    config.gamma = config.gamma if FLAGS.gamma is None else FLAGS.gamma
    config.beta = config.beta if FLAGS.beta is None else FLAGS.beta
    
    
    if FLAGS.load_weight:
        config.load_path = f"weights/{FLAGS.env}/"
        
    if FLAGS.save_weight:
        Path(f"weights/{FLAGS.env}").mkdir(parents=True, exist_ok=True)
        config.save = True
        config.save_path = f"weights/{FLAGS.env}/"
        
    return config
        
