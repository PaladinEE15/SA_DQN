import myutils


env_set = ["BankHeistNoFrameskip-v4","FreewayNoFrameskip-v4","PongNoFrameskip-v4","RoadRunnerNoFrameskip-v4"]


myutils.robust_learn(env_id=1, total_steps=40000, train_attack_mag=1/255, lr=0.0003, stb_path='buffers/FW_buffer', src_path='models/Freeway-natural.model', tgt_path='rob_models/rob_FW.pkl', log_name='Logs/FW', batch_size=32, robust_factor=1)
print('FW rob train done')
myutils.robust_learn(env_id=2, total_steps=40000, train_attack_mag=1/255, lr=0.0003, stb_path='buffers/PO_buffer', src_path='models/Pong-natural.model', tgt_path='rob_models/rob_PO.pkl', log_name='Logs/PO', batch_size=32, robust_factor=1)
print('PO rob train done')
myutils.robust_learn(env_id=3, total_steps=40000, train_attack_mag=1/255, lr=0.0003, stb_path='buffers/RR_buffer', src_path='models/RoadRunner-natural.model', tgt_path='rob_models/rob_RR.pkl', log_name='Logs/RR', batch_size=32, robust_factor=1)
print('RR rob train done')


print('BH raw atk:')
myutils.test_performance(target_model_config={'model_path':'models/BankHeist-natural.model'}, config_path='config/BankHeist_nat.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('BH cov atk:')
myutils.test_performance(target_model_config={'model_path':'models/BankHeist-convex.model'}, config_path='config/BankHeist_cov.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('BH myrob atk:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_BH.pkl'}, config_path='config/BankHeist_cov.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)

print('BH raw raw:')
myutils.test_performance(target_model_config={'model_path':'models/BankHeist-natural.model'}, config_path='config/BankHeist_nat.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('BH cov raw:')
myutils.test_performance(target_model_config={'model_path':'models/BankHeist-convex.model'}, config_path='config/BankHeist_cov.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('BH myrob raw:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_BH.pkl'}, config_path='config/BankHeist_cov.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)

print('FW raw atk:')
myutils.test_performance(target_model_config={'model_path':'models/Freeway-natural.model'}, config_path='config/Freeway.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('FW cov atk:')
myutils.test_performance(target_model_config={'model_path':'models/Freeway-convex.model'}, config_path='config/Freeway.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('FW myrob atk:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_FW.pkl'}, config_path='config/Freeway.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)

print('FW raw raw:')
myutils.test_performance(target_model_config={'model_path':'models/Freeway-natural.model'}, config_path='config/Freeway.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('FW cov raw:')
myutils.test_performance(target_model_config={'model_path':'models/Freeway-convex.model'}, config_path='config/Freeway.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('FW myrob raw:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_FW.pkl'}, config_path='config/Freeway.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)

print('PO raw atk:')
myutils.test_performance(target_model_config={'model_path':'models/Pong-natural.model'}, config_path='config/Pong.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('PO cov atk:')
myutils.test_performance(target_model_config={'model_path':'models/Pong-convex.model'}, config_path='config/Pong.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('PO myrob atk:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_PO.pkl'}, config_path='config/Pong.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)

print('PO raw raw:')
myutils.test_performance(target_model_config={'model_path':'models/Pong-natural.model'}, config_path='config/Pong.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('PO cov raw:')
myutils.test_performance(target_model_config={'model_path':'models/Pong-convex.model'}, config_path='config/Pong.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('PO myrob raw:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_PO.pkl'}, config_path='config/Pong.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)

print('RR raw atk:')
myutils.test_performance(target_model_config={'model_path':'models/RoadRunner-natural.model'}, config_path='config/RoadRunner.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('RR cov atk:')
myutils.test_performance(target_model_config={'model_path':'models/RoadRunner-convex.model'}, config_path='config/RoadRunner.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('RR myrob atk:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_RR.pkl'}, config_path='config/RoadRunner.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)

print('RR raw raw:')
myutils.test_performance(target_model_config={'model_path':'models/RoadRunner-natural.model'}, config_path='config/RoadRunner.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('RR cov raw:')
myutils.test_performance(target_model_config={'model_path':'models/RoadRunner-convex.model'}, config_path='config/RoadRunner.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('RR myrob raw:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_RR.pkl'}, config_path='config/RoadRunner.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)

'''
myutils.test_performance(target_model_config={'model_path':'models/Freeway-convex.model'}, config_path='config/Freeway_cov.json', is_attack=True, episode_num=10, max_frames=10000, compare_action=True, log_path=None, extra_model_config={'model_path':'models/Freeway-natural.model'})
myutils.test_performance(target_model_config={'model_path':'models/Pong-convex.model'}, config_path='config/Pong_cov.json', is_attack=True, episode_num=10, max_frames=10000, compare_action=True, log_path=None, extra_model_config={'model_path':'models/Pong-natural.model'})
myutils.test_performance(target_model_config={'model_path':'models/RoadRunner-convex.model'}, config_path='config/RoadRunner_cov.json', is_attack=True, episode_num=10, max_frames=10000, compare_action=True, log_path=None, extra_model_config={'model_path':'models/RoadRunner-natural.model'})


'''