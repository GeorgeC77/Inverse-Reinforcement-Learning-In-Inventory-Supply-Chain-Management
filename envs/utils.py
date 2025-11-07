



def Action_adapter(a, mu):
    # from [-1,1] to [0,max]
    max_action = 2 * mu
    return a * max_action / 2 + max_action / 2


def Action_adapter_reverse(act, mu):
    # from [0,max] to [-1,1]
    max_action = 2 * mu
    return (act - max_action / 2) / (max_action / 2)