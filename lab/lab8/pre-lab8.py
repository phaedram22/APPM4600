def eval_line(f, x0, x1, alpha):
    if x0 == x1:
        raise ValueError("x0 and x1 must be distinct.")

    f0 = f(x0)
    f1 = f(x1)

    # slope of the line through the two points
    slope = (f1 - f0) / (x1 - x0)

    # point-slope form: y = f0 + slope * (alpha - x0)
    return f0 + slope * (alpha - x0)