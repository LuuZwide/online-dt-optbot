import torch
#import d4rl 
import numpy as np
import pickle
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.Colab import ChartEnv, build
import matplotlib.pyplot as plt
import seaborn as sns
import gym
import numpy as np
import torch
import wandb

env_charts, env_close_prices, env_dates, env_test_charts, env_close_test_prices, env_dates_test = build.build_charts()

def get_attention_weights(model, state, actions, rewards, target_return, timesteps):
    """
    Extracts attention weights from the model to see if it is 
    attending to the target_return token.
    """
    model.eval()
    
    # 1. Ensure the model is configured to return attention weights
    # Most implementations require setting output_attentions=True in the transformer config
    # If your specific model doesn't support this, you'll need to use a hook (see below).
    
    with torch.no_grad():
        # Ensure the inputs have batch and sequence dimensions
        state = state.reshape(1, -1, model.state_dim)
        actions = actions.reshape(1, -1, model.act_dim)
        rewards = rewards.reshape(1, -1)
        target_return = target_return.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # Forward pass: Assuming model.forward() returns (state_preds, action_preds, return_preds, attention_weights)
        _, _, _, attention_weights = model(
            states=state,
            actions=actions,
            rewards=rewards,
            returns_to_go=target_return,
            timesteps=timesteps,
            output_attentions=True # This flag is key
        )
    
    # 2. Analyze weights
    # attention_weights usually has shape (num_layers, batch_size, num_heads, sequence_length, sequence_length)
    # For a Decision Transformer, your sequence length is (K * 3) if you pack (R, S, A)
    
    # Check the last layer's attention (usually the most refined)
    last_layer_attn = attention_weights[-1] # [batch, heads, seq, seq]
    
    # Average over heads
    avg_attn = last_layer_attn.mean(dim=1).squeeze(0) # [seq, seq]
    
    # In a typical DT (R, S, A) sequence:
    # The 'return' tokens are usually at indices [0, 3, 6...] 
    # If the model is conditioned, these indices should show high attention 
    # towards the recent 'state' and 'action' tokens.
    
    return avg_attn

# --- Alternative: Using a Hook if model.forward() doesn't return weights ---
def get_attn_via_hook(model):
    attn_storage = {}
    
    def hook_fn(module, input, output):
        # Captures the attention output of the specific attention layer
        attn_storage['weights'] = output[1] 

    model.transformer.h[-1].attn.register_forward_hook(hook_fn)
    return attn_storage 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path ="/opt/decision-transformer-optbot/data/chart.pkl"
with open(dataset_path, 'rb') as f:
    trajectories = pickle.load(f)

states, traj_lens, returns = [], [], []
for traj in trajectories:
    states.append(traj['observations'])
    traj_lens.append(len(traj['observations']))
    returns.append(traj['rewards'].sum())

# Concatenate all observations into a single array for proper normalization
all_observations = np.concatenate(states, axis=0)
state_mean = torch.from_numpy(np.mean(all_observations, axis=0)).to(device=device, dtype=torch.float32)
state_std = torch.from_numpy(np.std(all_observations, axis=0) + 1e-6).to(device=device, dtype=torch.float32)
state_dim = 41
act_dim = 5
max_ep_len = 1440
scale = 1.
action_range = [
            float(-1.0) + 1e-6,
            float(1.0) - 1e-6,
        ]

model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        action_range=action_range,
        max_length=20,
        eval_context_length=5,
        max_ep_len=1440,
        hidden_size=512,
        n_layer=4,
        n_head=4,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        stochastic_policy=True,
        ordering=1,
        init_temperature=0.1,
        bos_token=None,
        eos_token=None,
        target_entropy=-act_dim
    ).to(device=device)
max_length = 20

checkpoint = torch.load('./exp/Test/model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

done = False
    
env = ChartEnv.ChartEnv(chart_dict = env_test_charts, close_prices= env_close_test_prices , symbols = ['EURUSD', 'GBPUSD','USDJPY','USDCHF','AUDUSD'],timesteps = 1, episode_length = 1440, recurrent= False, random_start=True, dates_dict= env_dates_test, noise_level=1e-5)

state, _ = env.reset()
num_envs = 1

states = (
    torch.from_numpy(state)
    .reshape(num_envs, state_dim)
    .to(device=device, dtype=torch.float32)
).reshape(num_envs, -1, state_dim)

actions = torch.zeros((num_envs, 0, act_dim), device=device, dtype=torch.float32)

rewards = torch.zeros((num_envs, 0, 1), device=device, dtype=torch.float32)

target_return = 2.0

ep_return = target_return

target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
    num_envs, -1, 1
)

timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
    num_envs, -1
)
t = 0
K = 20   
avg_actions = []

episode_return = np.zeros((num_envs, 1)).astype(float)
episode_length = np.full(num_envs, np.inf)

curr_value_sum = 0
port_value_sum = 0
# run for 10 episodes
action_array = []
avg_returns = []
for episode in range(1440):

    actions = torch.cat(
                    [
                        actions,
                        torch.zeros((num_envs, act_dim), device=device).reshape(num_envs, -1, act_dim),
                    ],
                    dim=1,
    )[:, -K:].detach()
    rewards = torch.cat(
                    [
                        rewards,
                        torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
                    ],
                    dim=1,
    )[:, -K:].detach()

    state_pred, action_dist, reward_pred = model.get_predictions(
                    (states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32), # type: ignore
                    timesteps.to(dtype=torch.long),
                    num_envs=1,
                )
    
    state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
    reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)
    action = action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
    action = torch.where(action > 0.5, torch.ones_like(action), torch.zeros_like(action))
    print(f"Episode: {episode+1}, Action: {action.detach().cpu().numpy()}, Reward: {reward_pred}")
    next_state, reward, trunc, done, info = env.step(action.detach().cpu().numpy())
    episode_return += reward
    actions[:, -1] = action
    avg_actions.append(action.detach().cpu().numpy()[0])

    next_state = (
                torch.from_numpy(next_state)
                .reshape(num_envs, 1, state_dim)
                .to(device=device, dtype=torch.float32)
    )

    states = torch.cat([states, next_state], dim=1)[:, -K:]

    reward_tensor = torch.as_tensor(reward, device=device, dtype=torch.float32).reshape(num_envs, 1)
    rewards[:, -1] = reward_tensor
    pred_return = target_return[:, -1] - (reward_tensor * scale)
    timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(num_envs, 1) * (t + 1),
                ],
                dim=1,
    )[:, -K:]
    target_return = torch.cat([target_return, pred_return.reshape(num_envs, -1, 1)], dim=1 )[:, -K:].detach()
    t += 1

    if done:
        break   
    
    avg_returns.append(episode_return)
    #average actions
    #avg_actions, std_actions = np.round(np.mean(avg_actions),2), np.round(np.std(avg_actions),2)

print(f"avg_actions over {episode+1} episodes: {np.round(np.mean(avg_actions,axis=0),2)} ")

#print(f"Average Return over 10 episodes: {np.mean(avg_returns)}")
#print("Current Value sum : ", curr_value_sum)
#print("Port Value sum : ", port_value_sum)
#plt.figure(figsize=(10, 8))
#sns.heatmap(avg_attn.cpu().numpy(), cmap='viridis')
#plt.title("Attention Heatmap")
#plt.show()