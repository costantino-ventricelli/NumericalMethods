"""
   Questa classe calcola il numero massimo, il minimo e la precisone dei tre tipi di rappresentazione floating point:
   -   A doppia precisione;
   -   A singola precisione;
   -   A mezza precisione.
"""


class RealMinMax:
    __BASE = 2
    __SIGN_BIT = 1
    __DOUBLE_CHARACTERISTIC = 11
    __SINGLE_CHARACTERISTIC = 8
    __HALF_CHARACTERISTIC = 5
    __DOUBLE_PRECISION = 64
    __SINGLE_PRECISION = 32
    __HALF_PRECISION = 16

    def __init__(self, bit):
        print("Calcolo valori per: ", bit)
        if bit == RealMinMax.__DOUBLE_PRECISION:
            self.__bit = bit
            self.__characteristic = RealMinMax.__DOUBLE_CHARACTERISTIC
        elif bit == RealMinMax.__SINGLE_PRECISION:
            self.__bit = bit
            self.__characteristic = RealMinMax.__SINGLE_CHARACTERISTIC
        elif bit == RealMinMax.__HALF_PRECISION:
            self.__bit = bit
            self.__characteristic = RealMinMax.__HALF_CHARACTERISTIC
        else:
            print("Valore inserito per il bit non valido")
            self.__bit = 0
            self.__characteristic = 0
        self.__max_value = 0.00
        self.__min_value = 0.00
        self.__mantissa = 0
        self.__epsilon = 0.00

    """
       Il calcolo dei valori reali di massimo e minimo inizia calcolando il numero di cifre che compongono la mantissa,
       successivamente vendono individuati i valori limite della caratteristica, lower_bound e upper_bound, per farlo
       viene prima calcolato l'intero range del sistema di numerazione, full_interval, al quale viene sottatro 3 per
       eliminare gli estremi, che vengono usati per individuare i numeri speciali come 0 e infinito, e un altro valore
       per includere lo 0 nella numerazione.
       Dopo aver individuato i limiti superiore e inferiore si procede con il calcolo:
       - max_value: il calcolo (BASE - BASE^(mantissa)) deriva dal calcolo teorico che prevede:
           RealMax = 1.111...111 + 0.000...001, dove 111...111 viene ripetuto t volte e 000.001 gli 0 sono ripetuti
           t - 1 volte.
           Questa uguaglianza viene poi impostata:
               BASE = RealMax + BASE^(-t) => RealMax = BASE - BASE^(-t);
       - min_value: il calcolo è molto più semplice e prevede:
           BASE^(lower_bound);
       - epsilon: viene presa la base ed elevata alla -mantissa per ottenere la distanza tra 1 e il primo sucessore
           rappresentabile: BASE^(-mantissa)
    """
    def calculate_max_min_epsilon(self):
        if self.__bit != 0:
            self.__bit = self.__bit - RealMinMax.__SIGN_BIT
            self.__mantissa = self.__bit - self.__characteristic
            full_interval = (RealMinMax.__BASE ** self.__characteristic) - 3
            lower_bound = (-1) * int(full_interval / 2)
            upper_bound = full_interval - (-lower_bound)
            self.__max_value = ((RealMinMax.__BASE - RealMinMax.__BASE ** (-self.__mantissa))
                                * RealMinMax.__BASE ** upper_bound)
            self.__min_value = RealMinMax.__BASE ** lower_bound
            self.__epsilon = RealMinMax.__BASE ** (-self.__mantissa)

    def get_real_max(self):
        return self.__max_value

    def get_real_min(self):
        return self.__min_value

    def get_real_epsilon(self):
        return self.__epsilon
