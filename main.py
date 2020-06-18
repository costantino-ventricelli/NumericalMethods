from exercises.epsilon_machine import EpsilonMachine
from exercises.real_min_max import RealMinMax


def epsilon_machine_calculus():
    epsilon_machine = EpsilonMachine()
    epsilon_machine.find_epsilon()
    epsilon_machine.find_precision()
    epsilon_machine.find_significant_digits()
    print("Epsilon machine: ", epsilon_machine.get_epsilon())
    print("Machine precision: ", epsilon_machine.get_precision())
    print("Significant digits: ", epsilon_machine.get_significant_digits())


def real_min_max_calculus():
    real_min_max = RealMinMax()
    real_min_max.calculate_double_precision()
    real_min_max.calculate_single_precision()
    real_min_max.calculate_half_precision()


def main():
    print("\nEpsilon machine\n")
    epsilon_machine_calculus()
    print("\nReal min & max machine\n")
    real_min_max_calculus()


if __name__ == "__main__":
    main()
