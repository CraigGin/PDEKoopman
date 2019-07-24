from deephyper.benchmark import HpProblem

Problem = HpProblem()

Problem.add_dim('learning_rate',(10**(-6),10**(-3)))
Problem.add_dim('L1_lam',(0,10**(-4)))
Problem.add_dim('L2_lam',(0,10**(-4)))
Problem.add_dim('num_encoder_weights',(1,8))
Problem.add_dim('initialization',['He','identity'])
Problem.add_dim('add_identity',(0,1))
Problem.add_dim('diag_L',(0,1))
Problem.add_dim('log_space',(0,1))

Problem.add_starting_point(
    learning_rate = 10**(-4),
    L1_lam = 0.0,
    L2_lam = 10**(-8),
    num_encoder_weights = 6,
    initialization = 'He',
    add_identity = 1,
    diag_L = 1,
    log_space = 0)  

if __name__ == '__main__':
    print(Problem)