from shutil import copyfile
import datetime
import os
import ntpath
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_params(weight_path, env, ppo):
    if weight_path == "":
        raise Exception("\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
    print("\n*****************\nRetraining from the checkpoint: \n{%s}"%(weight_path), "\n*****************")

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'
    
    checkpoint = torch.load(weight_path)
    # model parameters
    ppo.gpu_model.load_state_dict(checkpoint['model_architecture_state_dict'])
    # actor (policy_network)
    ppo.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    # critic (value_network)
    ppo.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    
    print("\n=================\nCompleted Loading Parameters of model and optimizers")
    print("\n\nReturning Iteration Number to start retraining!")

    return int(iteration_number)


def reload_tb(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    '''
    Tags: 
    ['Reward/ball distance', 'Reward/com', 'Reward/contact', 
    'Reward/end effector', 'Reward/energy efficiency', 'Reward/orientation', 
    'Reward/reward_sum', 'Reward/velocity', 'Reward/total']
    '''
    
    first_scalar_tag = event_acc.Tags()['scalars'][0]
    last_iter = event_acc.Scalars(first_scalar_tag)[-1].step
    
    print("\n\nlast time, the tensorboard ended with iteration: ", last_iter, "\n=================\n")
    # # set agent's iteration to (last_iter) and set previous total_cont
    # agent.iter = last_iter
    # agent._reloaded_prev_total_sample_count = int(total_samples / 0.000001)
    
    # print("Agent.iter starting from: ", agent.iter + 1)
    # print("===================================================")
    # print("\n\n")
    return