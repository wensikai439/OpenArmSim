"""Chat channels module with plugin architecture."""

from PhyAgentOS.channels.base import BaseChannel
from PhyAgentOS.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
