from typing import Dict
from market_context import MarketContext
from rl_wrapper import SimpleRLWrapper
from agents.trend_finder_agent    import TrendFinderAgent
from agents.volume_spike_agent    import VolumeSpikeAgent
from agents.volatility_cluster_agent import VolatilityClusterAgent
from agents.rsi_regime_agent      import RSIRegimeAgent
from agents.macd_divergence_agent import MACDDivergenceAgent
from agents.adx_strength_agent    import ADXStrengthAgent
from agents.liquidity_mapper_agent  import LiquidityMapperAgent
from agents.pnl_cluster_rebalancer_agent import PnLClusterRebalancerAgent
from agents.correlation_monitor_agent   import CorrelationMonitorAgent
from agents.news_sentiment_agent    import NewsSentimentAgent
from agents.funding_rate_agent      import FundingRateAgent
from agents.whale_wallet_agent      import WhaleWalletAgent
from agents.time_of_day_agent       import TimeOfDayAgent

AGENT_CLASSES = [
    TrendFinderAgent, VolumeSpikeAgent, VolatilityClusterAgent,
    RSIRegimeAgent, MACDDivergenceAgent, ADXStrengthAgent,
    LiquidityMapperAgent, PnLClusterRebalancerAgent,
    CorrelationMonitorAgent, NewsSentimentAgent,
    FundingRateAgent, WhaleWalletAgent, TimeOfDayAgent
]

class MetaAgentController:
    """
    Orchestrates all agents + market context + RL + collaboration boosting.
    """
    def __init__(self, use_rl=True):
        self.agents = [cls() for cls in AGENT_CLASSES]
        self.rl = SimpleRLWrapper() if use_rl else None

    def run_all(self, market_data) -> Dict[str, any]:
        context = MarketContext(market_data)
        signals = {}
        agent_signals = {}
        raw_weights = {}
        ensemble_votes = {"buy":0.0, "sell":0.0, "hold":0.0, "avoid":0.0}

        # 1) Each agent votes
        for agent in self.agents:
            agent.process_data(market_data.copy(), context=context)
            sig = agent.generate_signal()
            wr  = agent.get_win_rate()
            w   = wr * agent.confidence
            agent.update_memory(sig)
            signals[agent.name]       = sig
            agent_signals[agent.name] = sig
            raw_weights[agent.name]   = w

        # 2) Collaboration boost
        alignment = {k:0 for k in ensemble_votes}
        for sig in agent_signals.values():
            alignment[sig] += 1
        for agent in self.agents:
            base  = raw_weights[agent.name]
            boost = 1 + (alignment[signals[agent.name]] / len(self.agents))
            ensemble_votes[signals[agent.name]] += base * boost

        # 3) Final decision: RL or weighted ensemble
        if self.rl:
            state = self.rl.get_state(agent_signals, context.all())
            final = self.rl.select_action(state)
        else:
            final = max(ensemble_votes, key=ensemble_votes.get)

        signals["master_decision"] = final
        signals["weights"]         = ensemble_votes
        signals["context"]         = context.all()
        return signals

