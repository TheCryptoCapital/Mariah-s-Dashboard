def get_strategy_params(mode):
    if mode == "Scalping":
        return 0.3, 0.6, 70, 30
    elif mode == "Swing":
        return 1.0, 3.0, 65, 35
    elif mode == "Momentum":
        return 1.5, 5.0, 60, 40
    else:
        return 1.0, 2.0, 65, 35  # default fallback

