
def get_strategy_params(mode="Swing"):
    if mode == "Scalp":
        return 0.015, 0.03, 80, 20
    elif mode == "Momentum":
        return 0.02, 0.04, 85, 15
    return 0.01, 0.02, 70, 30
