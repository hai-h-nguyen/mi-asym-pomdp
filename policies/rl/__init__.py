from .dqn import DQN
from .dqn_zp import DQN_ZP
from .dqn_believer import DQN_BELIEVER
from .dqn_believer_rnn import DQN_BELIEVER_RNN
from .dqn_ours import DQN_OURS
from .dqn_on_off import DQN_On_Off
from .dqn_ua import DQN_UA
from .dqn_ba import DQN_BA

from .sac import SAC
from .sac_zp import SAC_ZP
from .sac_believer import SAC_BELIEVER
from .sac_believer_rnn import SAC_BELIEVER_RNN
from .sac_ours import SAC_OURS
from .sac_embed import SAC_Embed
from .sac_on_off import SAC_On_Off
from .sac_ua import SAC_UA
from .sac_ba import SAC_BA

RL_ALGORITHMS = {
    DQN.name: DQN,
    DQN_ZP.name: DQN_ZP,
    DQN_BELIEVER.name: DQN_BELIEVER,
    DQN_BELIEVER_RNN.name: DQN_BELIEVER_RNN,
    DQN_OURS.name: DQN_OURS,
    DQN_On_Off.name: DQN_On_Off,
    DQN_UA.name: DQN_UA,
    DQN_BA.name: DQN_BA,

    SAC.name: SAC,
    SAC_ZP.name: SAC_ZP,
    SAC_BELIEVER.name: SAC_BELIEVER,
    SAC_OURS.name: SAC_OURS,
    SAC_Embed.name: SAC_Embed,
    SAC_On_Off.name: SAC_On_Off,
    SAC_UA.name: SAC_UA,
    SAC_BA.name: SAC_BA,
}
