from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn


class AgentInterface(nn.Module):
    """
    Abstract interface describing the methods an RL agent must provide
    to be used by selfplay.

    Concrete implementations must accept torch.Tensor inputs and return
    torch.Tensor outputs with the same semantics as the Agent in ppoBC_PPO.py.
    """

    @abstractmethod
    def __init__(self, mapsize: int, lstm_hidden: int, lstm_layers: int) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, sc: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute and return the flattened feature vector used by actor & critic.
        Args:
            x: observation tensor (e.g. [B, H, W, C])
            sc: scalar inputs tensor
            z: Expert embedding tensor
        Returns:
            concatenated feature tensor
        """
        raise NotImplementedError

    @abstractmethod
    def get_action(
        self,
        x: torch.Tensor,
        Sc: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        invalid_action_masks: Optional[torch.Tensor] = None,
        envs: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Produce actions for the given observations.

        Returns a tuple:
            action: chosen actions (tensor)
            logprob: log-probability per environment (tensor)
            entropy: entropy per environment (tensor)
            invalid_action_masks (tensor)
        """
        raise NotImplementedError

    @abstractmethod
    def get_value(self, x: torch.Tensor, Sc: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Critic value prediction for given observations.
        """
        raise NotImplementedError