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
    array_of_bit = [64, 32, 16, 0]
    for item in array_of_bit:
        double_precision = RealMinMax(item)
        double_precision.calculate_max_min_epsilon()
        print("Real max: ", double_precision.get_real_max())
        print("Real min: ", double_precision.get_real_min())
        print("Epsilon: ", double_precision.get_real_epsilon(), "\n")


def main():
    print("\nEpsilon machine\n")
    epsilon_machine_calculus()
    print("\nReal min & max machine\n")
    real_min_max_calculus()


if __name__ == "__main__":
    main()
