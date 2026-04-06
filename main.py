"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

# type: ignore
import argparse
import pickle
import random
import time
import gym
import torch
import numpy as np
import wandb

import utils
from replay_buffer import ReplayBuffer
from lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer
from logger import Logger
from decision_transformer.Colab import build
import os

MAX_EPISODE_LEN = 1440
os.environ["WANDB_MODE"] = "offline"


class Experiment:
    def __init__(self, variant):
        checkpoint = None
        if variant.get("continue_training"):
            checkpoint = self._get_checkpoint(variant["model_path_prefix"])
            variant["log_path"] = checkpoint.get("log_path")

        self.variant = variant

        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
            variant["env"]
        )
        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)

        self.aug_trajs = []

        self.device = variant.get("device", "cuda")
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
        ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(# type: ignore
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999], # type: ignore
        )

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.reward_scale = 1.0
        variant["exp_name"] = "ODT-"+ variant["env"]+ "-"+ variant["tag"]

        wandb_kwargs = {
            "name": variant["exp_name"],
            "project": "decision-transformer-opt-experiments",
            "config": variant,
            "tags": [],
            "reinit": True,
        }
        if checkpoint is not None:
            wandb_run_id = checkpoint.get("wandb_run_id")
            if not wandb_run_id:
                raise ValueError("Checkpoint is missing wandb_run_id; cannot resume the same W&B run.")
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "must"

        wandb.init(**wandb_kwargs)
        print("wandb initialised")
        self.logger = Logger(variant)

        if checkpoint is not None:
            self._restore_checkpoint(checkpoint, variant["model_path_prefix"])

    def _get_env_spec(self, variant):
        state_dim = 26
        act_dim = 4
        action_range = [
            float(-1.0) + 1e-6,
            float(1.0) - 1e-6,
        ]
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
            "replay_buffer_trajectories": self.replay_buffer.trajectories,
            "replay_buffer_start_idx": self.replay_buffer.start_idx,
            "aug_trajs": self.aug_trajs,
            "log_path": self.logger.log_path,
            "wandb_run_id": wandb.run.id if wandb.run is not None else None,
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _get_checkpoint(self, path_prefix):
        checkpoint_path = Path(path_prefix) / "model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            return torch.load(f)

    def _restore_checkpoint(self, checkpoint, path_prefix):
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.log_temperature_optimizer.load_state_dict(
            checkpoint["log_temperature_optimizer_state_dict"]
        )
        self.pretrain_iter = checkpoint["pretrain_iter"]
        self.online_iter = checkpoint["online_iter"]
        self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
        self.replay_buffer = ReplayBuffer(
            self.variant["replay_size"],
            checkpoint.get("replay_buffer_trajectories", self.offline_trajs),
        )
        self.replay_buffer.start_idx = checkpoint.get("replay_buffer_start_idx", 0)
        self.aug_trajs = checkpoint.get("aug_trajs", [])
        np.random.set_state(checkpoint["np"])
        random.setstate(checkpoint["python"])
        torch.set_rng_state(checkpoint["pytorch"])
        print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):

        dataset_path = f"./data/{env_name}.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std

    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN

        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * online_envs.num_envs

            if randomized:
                target_return = [
                    random.uniform(0, target_explore) * self.reward_scale
                    for _ in range(online_envs.num_envs)
                ]

            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
            )

        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }

    def pretrain(self, eval_envs, loss_fn):
        print("\n\n\n*** Pretrain ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                env_name=self.variant["env"],
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            eval_outputs, eval_reward = self.evaluate(eval_fns)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            rcl_outputs = self.evaluate_rcsl(eval_envs)
            outputs.update(rcl_outputs)
            
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
            )

            self.pretrain_iter += 1

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def online_tuning(self, online_envs, eval_envs, loss_fn):

        print("\n\n\n*** Online Finetuning ***")

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        eval_fns = [
            create_vec_eval_episodes_fn(
                env_name=self.variant["env"],
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        
        while self.online_iter < self.variant["max_online_iters"]:
            
            outputs = {}
            augment_outputs = self._augment_trajectories(
                online_envs,
                self.variant["online_rtg"],
                n=self.variant["num_online_rollouts"],
                randomized=self.variant['randomized_target_return']
            )
            outputs.update(augment_outputs)

            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            outputs.update(train_outputs)

            if evaluation:
                eval_outputs, eval_reward = self.evaluate(eval_fns)
                outputs.update(eval_outputs)

                rcl_outputs = self.evaluate_rcsl(eval_envs)
                outputs.update(rcl_outputs)

            outputs["time/total"] = time.time() - self.start_time

            # log the metrics
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
            )

            self.online_iter += 1

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=False,
            )

    def evaluate_rcsl(self, eval_envs):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        rcsl_table = wandb.Table(columns=["Target Performance", "Actual Performance"], allow_mixed_types=True) #Normalised scores 
        rcsl_error_table = wandb.Table(columns=["Target Return", "MSE"], allow_mixed_types=True) #Mean Squared Error(MSE) L2
        rcsl_std_table = wandb.Table(columns=["Target Return", "STD"], allow_mixed_types=True) #Standard Deviation(STD)

        rcsl_mean_length = wandb.Table(columns=["Target Return", "Mean Length"], allow_mixed_types=True) #Mean Length of episodes
        rcsl_std_length = wandb.Table(columns=["Target Return", "STD Length"], allow_mixed_types=True) #STD of Length of episodes

        rc_loss = 0

        for eval_rtg_coef in [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]:
            
            eval_rtg = self.variant["eval_rtg"] * eval_rtg_coef
            print(f"Evaluating on eval_rtg_coef: {eval_rtg_coef}, ({eval_rtg})")
            
            eval_fns = [
                create_vec_eval_episodes_fn(
                    env_name=self.variant["env"],
                    vec_env=eval_envs,
                    eval_rtg=eval_rtg,
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    device=self.device,
                    use_mean=True,
                    reward_scale=self.reward_scale,
                )
            ] * int(100 / self.variant["num_eval_rollouts"])
            score_mean_gms = []
            lengths_gm = []
            std_lengths = []
            for eval_fn in eval_fns:
                o = eval_fn(self.model)
                score_mean_gms.append(o["evaluation/score_mean_gm"])   
                lengths_gm.append(o["evaluation/length_mean_gm"])
                std_lengths.append(o["evaluation/length_std_gm"])

            mean_scores, std_scores = np.mean(score_mean_gms), np.std(score_mean_gms)       
            mean_length, std_length = np.mean(lengths_gm), np.std(std_lengths)
            
            target_performance = eval_rtg
            rc_error = (target_performance - mean_scores)**2

            rcsl_table.add_data(target_performance, mean_scores)
            rcsl_error_table.add_data(eval_rtg, rc_error)
            
            rcsl_std_table.add_data(eval_rtg, std_scores)
            rcsl_mean_length.add_data(eval_rtg, mean_length)
            rcsl_std_length.add_data(eval_rtg, std_length)

            rc_loss += rc_error    
        
        step = self.pretrain_iter + self.online_iter
        outputs['rcsl_evaluation/RCSL Error Table'] = rcsl_error_table
        outputs["rcsl_evaluation/RCSL Table"] = rcsl_table
        outputs['rcsl_evaluation/RCSL std Table'] = rcsl_std_table 
        outputs["rcsl_evaluation/RCSL total loss"] = rc_loss
        outputs["rcsl_evaluation/RCSL mean length"] = rcsl_mean_length
        outputs["rcsl_evaluation/RCSL std length"] = rcsl_std_length

        return outputs

    def __call__(self):

        utils.set_seed_everywhere(args.seed)

        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

            entropy = a_hat_dist.entropy().mean()
            loss = -(log_likelihood + entropy_reg * entropy)

            return (
                loss,
                -log_likelihood,
                entropy,
            )

        def get_env_builder(type = 0):
            def make_env_fn():

                env, eval_env = build.build_env()
                if type == 0:
                    return env
                else:
                    return eval_env

            return make_env_fn

        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        target_goal = None
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(type = 1)
                for i in range(self.variant["num_eval_episodes"])
            ] # type: ignore
        )

        self.start_time = time.time()

        if self.variant["continue_training"]:
            print(f"Resuming training from {self.variant['model_path_prefix']}")



        if self.variant["max_pretrain_iters"]:
            self.pretrain(eval_envs, loss_fn)

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(type = 0)
                    for i in range(self.variant["num_online_rollouts"])
                ] # type: ignore
            )
            self.online_tuning(online_envs, eval_envs, loss_fn)
            online_envs.close()

        eval_envs.close()
        wandb.finish()
        print(f"wandb finialized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--env", type=str, default="hopper-medium-v2")

    # model options
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=5)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=float, default=3600)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--num_eval_rollouts", type=int, default=10)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=1)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=5000)

    # finetuning options
    parser.add_argument("--max_online_iters", type=int, default=1500)
    parser.add_argument("--online_rtg", type=float, default=7200)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=10)

    # environment options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")

    parser.add_argument("--randomized_target_return", type=bool, default=False)
    parser.add_argument("--tag" , type=str, default="")
    parser.add_argument("--continue_training", type=bool, default=False)
    parser.add_argument("--model_path_prefix", type=str, default="./exp/default")

    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args))

    print("=" * 50)
    experiment()
