from .policy_mlp import ModelFreeOffPolicy_MLP as Policy_MLP
from .policy_mlp_embed import ModelFreeOffPolicy_MLP_Embed as Policy_MLP_Embed
from .policy_mlp_on_off import ModelFreeOffPolicy_MLP_On_Off as Policy_MLP_On_Off
from .policy_mlp_dqn import ModelFreeOffPolicy_DQN_MLP as Policy_MLP_DQN
from .policy_mlp_dqn_on_off import ModelFreeOffPolicy_DQN_MLP_On_Off as Policy_MLP_DQN_On_Off

from .policy_rnn import ModelFreeOffPolicy_Separate_RNN as Policy_Separate_RNN
from .policy_rnn_zp import ModelFreeOffPolicy_Separate_RNN_ZP as Policy_Separate_RNN_ZP
from .policy_rnn_believer_shared import ModelFreeOffPolicy_Shared_RNN_Believer as Policy_Shared_RNN_Believer
from .policy_rnn_believer_gpt_shared import ModelFreeOffPolicy_Shared_GPT_Believer as Policy_Shared_GPT_Believer
from .policy_rnn_ours import ModelFreeOffPolicy_Separate_RNN_Ours as Policy_Separate_RNN_Ours
from .policy_rnn_ua import ModelFreeOffPolicy_Separate_RNN_UA as Policy_Separate_RNN_UA
from .policy_rnn_ba import ModelFreeOffPolicy_Separate_RNN_BA as Policy_Separate_RNN_BA

from .policy_rnn_dqn import ModelFreeOffPolicy_DQN_RNN as Policy_DQN_RNN
from .policy_rnn_dqn_ours import ModelFreeOffPolicy_DQN_RNN_Ours as Policy_DQN_RNN_Ours
from .policy_rnn_dqn_zp import ModelFreeOffPolicy_DQN_RNN_ZP as Policy_DQN_RNN_ZP
from .policy_rnn_dqn_believer import ModelFreeOffPolicy_DQN_RNN_Believer as Policy_DQN_RNN_Believer
from .policy_rnn_dqn_believer_gpt import ModelFreeOffPolicy_DQN_GPT_Believer as Policy_DQN_GPT_Believer
from .policy_rnn_dqn_ua import ModelFreeOffPolicy_DQN_RNN_UA as Policy_DQN_RNN_UA
from .policy_rnn_dqn_ba import ModelFreeOffPolicy_DQN_RNN_BA as Policy_DQN_RNN_BA

AGENT_CLASSES = {
    "Policy_MLP": Policy_MLP,
    "Policy_MLP_Embed": Policy_MLP_Embed,
    "Policy_MLP_On_Off": Policy_MLP_On_Off,
    "Policy_DQN_MLP": Policy_MLP_DQN,
    "Policy_DQN_MLP_On_Off": Policy_MLP_DQN_On_Off,

    "Policy_Separate_RNN": Policy_Separate_RNN,
    "Policy_Separate_RNN_Ours": Policy_Separate_RNN_Ours,
    "Policy_Separate_RNN_ZP": Policy_Separate_RNN_ZP,
    "Policy_Shared_RNN_Believer": Policy_Shared_RNN_Believer,
    "Policy_Shared_GPT_Believer": Policy_Shared_GPT_Believer,
    "Policy_Separate_RNN_UA": Policy_Separate_RNN_UA,
    "Policy_Separate_RNN_BA": Policy_Separate_RNN_BA,

    "Policy_DQN_RNN": Policy_DQN_RNN,
    "Policy_DQN_RNN_ZP": Policy_DQN_RNN_ZP,
    "Policy_DQN_RNN_Believer": Policy_DQN_RNN_Believer,
    "Policy_DQN_GPT_Believer": Policy_DQN_GPT_Believer,
    "Policy_DQN_RNN_Ours": Policy_DQN_RNN_Ours,
    "Policy_DQN_RNN_UA": Policy_DQN_RNN_UA,
    "Policy_DQN_RNN_BA": Policy_DQN_RNN_BA,
}


assert Policy_Separate_RNN.ARCH == Policy_DQN_RNN.ARCH

from enum import Enum


class AGENT_ARCHS(str, Enum):
    # inherit from str to allow comparison with str
    Markov = Policy_MLP.ARCH
    Memory = Policy_Separate_RNN.ARCH
