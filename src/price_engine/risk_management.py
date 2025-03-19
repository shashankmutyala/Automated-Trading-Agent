def manage_risk(position, max_loss=0.02, current_price=None, entry_price=None):
    """Risk management logic for open positions."""
    loss = (entry_price - current_price) / entry_price
    if loss > max_loss:
        return "CLOSE_POSITION"
    return "HOLD"
