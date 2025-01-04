import numpy
import torch
import torch
from torch.distributions.normal import Normal


class ReprTrainer():
    def __init__(self, env, exps, rep_model, train_config):

        self.env = env
        self.exps = exps
        self.rep_model = rep_model

        self.beta = train_config.beta
        self.dynamics_loss_s_coef = train_config.dynamics_loss_s_coef
        self.dynamics_loss_o_coef = train_config.dynamics_loss_o_coef
        self.reward_loss_coef = train_config.reward_loss_coef
        self.batch_size = train_config.batch_size

        adam_eps = train_config.adam_eps
        lr = train_config.lr
        self.optimizer = torch.optim.Adam(self.rep_model.parameters(),
                                          lr,
                                          eps=adam_eps)
        self.batch_num = 0
        self.num_frames = self.exps.next_mask.shape[0]

        self.use_mi_estimator = False
        mi_lr = train_config.mi_lr
        self.mi_optimizer = torch.optim.Adam(self.mi_estimator.parameters(),
                                                 mi_lr,
                                                 eps=adam_eps)

    def update(self):
        log_losses = []
        log_state_dynamics_losses = []
        log_obs_dynamics_losses = []
        log_reward_losses = []
        log_kl_losses = []
        log_grad_norms = []

        for inds in self._get_batches_idxes():
            # Initialize batch values
            batch_loss = 0
            batch_state_dynamics_loss = 0
            batch_obs_dynamics_loss = 0
            batch_reward_loss = 0
            batch_kl_loss = 0

            # Create a sub-batch of experience
            sb = self.exps[inds]

            # Compute loss
            encoder_mean_s, encoder_std_s = self.rep_model.encode_state(sb.state)

            next_state_preds, next_obs_preds, reward_preds = self.rep_model.predict_next(sb.state, sb.obs, sb.action)
            next_state_targets, _ = self.rep_model.encode_state(sb.next_state)
            next_obs_targets, _ = self.rep_model.encode_obs(sb.next_obs)

            state_dynamics_loss = torch.pow(torch.norm(next_state_preds - (next_state_targets.detach() * sb.next_mask.unsqueeze(dim=1)), p=2, dim=1), 2).mean()
            obs_dynamics_loss = torch.pow(torch.norm(next_obs_preds - (next_obs_targets.detach() * sb.next_mask.unsqueeze(dim=1)), p=2, dim=1), 2).mean()
            reward_loss = torch.pow(sb.reward.unsqueeze(dim=1) - reward_preds, 2).mean()

            # This is E_s[KL(p(Z|S)|| q(Z))] = E_S[E_Z[log(p(Z|S)||q(Z))]]
            kl_loss = torch.distributions.kl.kl_divergence(Normal(encoder_mean_s, encoder_std_s),
                                      Normal(torch.zeros_like(encoder_mean_s),
                                             torch.ones_like(encoder_mean_s)))

            loss = self.beta * kl_loss
            loss += self.dynamics_loss_s_coef * state_dynamics_loss
            loss += self.dynamics_loss_o_coef * obs_dynamics_loss
            loss += self.reward_loss_coef * reward_loss

            # Update batch values
            batch_loss += loss.mean()
            batch_state_dynamics_loss += state_dynamics_loss.item()
            batch_obs_dynamics_loss += obs_dynamics_loss.item()
            batch_reward_loss += reward_loss.item()
            batch_kl_loss += kl_loss.mean().item()

            self.optimizer.zero_grad()
            batch_loss.backward()
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.rep_model.parameters() if p.grad is not None) ** 0.5
            self.optimizer.step()

            # Update log values
            log_losses.append(batch_loss.item())
            log_state_dynamics_losses.append(batch_state_dynamics_loss)
            log_obs_dynamics_losses.append(batch_obs_dynamics_loss)
            log_reward_losses.append(batch_reward_loss)
            log_kl_losses.append(batch_kl_loss)
            log_grad_norms.append(grad_norm)

        torch.set_printoptions(sci_mode=False)
        print("Actual rewards:", [round(x.item(), 3) for x in list(sb.reward[:10])])
        print("Predicted rewards:", [round(x.item(), 3) for x in list(reward_preds[:10])])

        logs = {
            "grad_norm": numpy.mean(log_grad_norms),
            "state_dynamics_loss": numpy.mean(log_state_dynamics_losses),
            "obs_dynamics_loss": numpy.mean(log_obs_dynamics_losses),
            "reward_loss": numpy.mean(log_reward_losses),
            "kl_loss": numpy.mean(log_kl_losses)
        }

        return logs

    def update_simple(self):
        log_losses = []
        log_state_dynamics_losses = []
        log_reward_losses = []
        log_grad_norms = []

        for inds in self._get_batches_idxes():
            # Initialize batch values
            batch_loss = 0
            batch_state_dynamics_loss = 0
            batch_reward_loss = 0

            # Create a sub-batch of experience
            sb = self.exps[inds]

            # Compute loss
            next_state_preds, _, reward_preds = self.rep_model.predict_next(sb.state, sb.obs, sb.action)
            next_state_targets, _ = self.rep_model.encode_state(sb.next_state)

            state_dynamics_loss = torch.pow(torch.norm(next_state_preds - (next_state_targets.detach() * sb.next_mask.unsqueeze(dim=1)), p=2, dim=1), 2).mean()
            reward_loss = torch.pow(sb.reward.unsqueeze(dim=1) - reward_preds, 2).mean()

            loss = self.dynamics_loss_s_coef * state_dynamics_loss
            loss += self.reward_loss_coef * reward_loss

            # Update batch values
            batch_loss += loss.mean()
            batch_state_dynamics_loss += state_dynamics_loss.item()
            batch_reward_loss += reward_loss.item()

            self.optimizer.zero_grad()
            batch_loss.backward()
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.rep_model.parameters() if p.grad is not None) ** 0.5
            self.optimizer.step()

            # Update log values
            log_losses.append(batch_loss.item())
            log_state_dynamics_losses.append(batch_state_dynamics_loss)
            log_reward_losses.append(batch_reward_loss)
            log_grad_norms.append(grad_norm)

        torch.set_printoptions(sci_mode=False)
        print("Actual rewards:", [round(x.item(), 3) for x in list(sb.reward[:10])])
        print("Predicted rewards:", [round(x.item(), 3) for x in list(reward_preds[:10])])

        logs = {
            "grad_norm": numpy.mean(log_grad_norms),
            "state_dynamics_loss": numpy.mean(log_state_dynamics_losses),
            "reward_loss": numpy.mean(log_reward_losses),
        }

        return logs

    def _get_batches_idxes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames)
        indexes = numpy.random.permutation(indexes)

        self.batch_num += 1

        num_indexes = self.batch_size
        batches_starting_indexes = [indexes[i:i+num_indexes]
                                    for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes