from ladder import *


def test_harmonic_ladder(plot=False):
    # Create an harmonic potential
    v = Potential(Grid(-10, 10, 1001))
    v.values = v.x.values ** 2

    # Diagonalize
    spec = v.calculate_eigenstates()

    target = spec.orbitals[:10]

    # Generate ladder states
    ladder = HarmonicLadderOperator()
    ladder_derivative = DerivativeLadderOperator()
    from_previous = [spec.orbitals[0]]
    recursive = [spec.orbitals[0]]
    derivative = [spec.orbitals[0]]
    while len(from_previous) < len(target):
        # apply ladder to previous analytic state, to avoid accumulation of errors
        from_previous.append(ladder.apply(target[len(from_previous) - 1], None))

        # apply ladder operator recursively
        recursive.append(ladder.apply(recursive[-1], None))

        # Get simple derivatives for comparison
        derivative.append(ladder_derivative.apply(derivative[-1], None))

    if plot:
        import matplotlib.pyplot as plt
        for i in range(len(target)):
            a, b, c, d = target[i], from_previous[i], recursive[i], derivative[i]
            plt.plot(a.x.values, a.values + i, color="green")
            plt.plot(b.x.values, b.values + i, color="blue")
            plt.plot(c.x.values, c.values + i, color="red")
            plt.plot(d.x.values, d.values + i, color="black", alpha=0.1)
        plt.show()

    for a, b in zip(target, from_previous):
        assert_all_close(a.values, b.values, atol=0.1, message=f"Orbitals not reproduced by ladder operator!")
