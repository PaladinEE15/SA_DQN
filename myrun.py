import myutils


env_set = ["BankHeistNoFrameskip-v4","FreewayNoFrameskip-v4","PongNoFrameskip-v4","RoadRunnerNoFrameskip-v4"]
env_paraset = [{"restrict_actions": True},{"crop_shift": 0},{"crop_shift": 10,      "restrict_actions": 4},{"crop_shift": 20, "restrict_actions": True}]
model_paths = []

myutils.test_performance(env_name=env_set[0], target_model_config={'model_path':'models/BankHeist-convex.model'}, env_params=env_paraset[0], attack_config={'method':'pgd', 'is_atk':True}, episode_num=5, max_frames=10000, compare_action=True, log_path=None, extra_model_config={'model_path':'models/BankHeist-natural.model'})