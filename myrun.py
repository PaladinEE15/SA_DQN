import myutils
import sys
#paras = int(sys.argv[1])
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
env_set = ["BankHeistNoFrameskip-v4","FreewayNoFrameskip-v4","PongNoFrameskip-v4","RoadRunnerNoFrameskip-v4"]

#myutils.robust_learn(env_id=0, total_steps=40000, train_attack_mag=1/255, lr=0.0003, stb_path="buffers/BH_buffer", src_path="models/BankHeist-natural.model", tgt_path="rob_models/rob_BH_cov.pkl", is_cov=True, log_name=None, batch_size=32, robust_factor=1)
myutils.rob_train_and_test(env_id=0, robust_factor_and_steps=[[1,1],[0.5,1]],model_id=['10','11'],is_cov=True)
'''
if paras == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    myutils.rob_train_and_test(env_id=4, robust_factor_and_steps=[[1,1],[0.5,1],[0.5,5]],model_id=['1','2','3'])
elif paras == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    myutils.rob_train_and_test(env_id=5, robust_factor_and_steps=[[1,1],[0.5,1],[0.5,5]],model_id=['1','2','3'])
elif paras == 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    myutils.rob_train_and_test(env_id=6, robust_factor_and_steps=[[1,1],[0.5,1],[0.5,5]],model_id=['1','2','3'])
elif paras == 3:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    myutils.rob_train_and_test(env_id=7, robust_factor_and_steps=[[1,1],[0.5,1],[0.5,5]],model_id=['1','2','3'])


print('FW raw atk:')
myutils.test_performance(target_model_config={'model_path':'models/Freeway-natural.model'}, config_path='config/Freeway_nat.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('FW cov atk:')
myutils.test_performance(target_model_config={'model_path':'models/Freeway-convex.model'}, config_path='config/Freeway_nat.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('FW myrob atk:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_FW.pkl'}, config_path='config/Freeway_nat.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)

print('FW raw raw:')
myutils.test_performance(target_model_config={'model_path':'models/Freeway-natural.model'}, config_path='config/Freeway_nat.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('FW cov raw:')
myutils.test_performance(target_model_config={'model_path':'models/Freeway-convex.model'}, config_path='config/Freeway_nat.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('FW myrob raw:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_FW.pkl'}, config_path='config/Freeway_nat.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)


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



print('PO raw atk:')
myutils.test_performance(target_model_config={'model_path':'models/Pong-natural.model'}, config_path='config/Pong_nat.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('PO cov atk:')
myutils.test_performance(target_model_config={'model_path':'models/Pong-convex.model'}, config_path='config/Pong_nat.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('PO myrob atk:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_PO.pkl'}, config_path='config/Pong_nat.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)

print('PO raw raw:')
myutils.test_performance(target_model_config={'model_path':'models/Pong-natural.model'}, config_path='config/Pong_nat.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('PO cov raw:')
myutils.test_performance(target_model_config={'model_path':'models/Pong-convex.model'}, config_path='config/Pong_nat.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('PO myrob raw:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_PO.pkl'}, config_path='config/Pong_nat.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)

print('RR raw atk:')
myutils.test_performance(target_model_config={'model_path':'models/RoadRunner-natural.model'}, config_path='config/RoadRunner_nat.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('RR cov atk:')
myutils.test_performance(target_model_config={'model_path':'models/RoadRunner-convex.model'}, config_path='config/RoadRunner_nat.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('RR myrob atk:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_RR.pkl'}, config_path='config/RoadRunner_nat.json', is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)

print('RR raw raw:')
myutils.test_performance(target_model_config={'model_path':'models/RoadRunner-natural.model'}, config_path='config/RoadRunner_nat.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('RR cov raw:')
myutils.test_performance(target_model_config={'model_path':'models/RoadRunner-convex.model'}, config_path='config/RoadRunner_nat.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
print('RR myrob raw:')
myutils.test_performance(target_model_config={'model_path':'rob_models/rob_RR.pkl'}, config_path='config/RoadRunner_nat.json', is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
'''
