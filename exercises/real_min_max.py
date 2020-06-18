#   Questa classe calcola il numero massimo, il minimo e la precisone dei tre tipi di rappresentazione floating point:
#   -   A doppia precisione;
#   -   A singola precisione;
#   -   A mezza precisione.


class RealMinMax:

    def __init__(self):
        print("Precision calculator")

    #   Il valore di RealMin ha come mantissa un valore pari a 1, in quanto é formato nel seguente modo:
    #   Rmin = 1.0000...000, dove gli zeri sono ripetuti 53 volte in questo caso, 53 in quanto nella rappresentazione
    #   normale dei numeri in floating point il primo valore è sempre uno, quindi dato questo come vero sempre si può
    #   iniziare a dare valore alle cifre della mantissa escluendo la prima, quindi il numero di cifre della mantissa
    #   diventa t + 1 => 52 + 1.
    #   Nel calcolo del RealMax invece avremo una situazione del genere 1.1111...111 dove gli 1 sono ripetuti 53 volte,
    #   raggiungendo un valore prossimo a 2, al quale aggiungendo 0.0000...001, dove gli 0 sono ripetuti t - 1 volte,
    #   ci darà 2 come risultato, quindi scriviamo:
    #   2 = 1.1111...111 + 0.0000...001 = RealMax + 2^(-t) =>
    #   => RealMax = 2 - 2^(-t).
    @staticmethod
    def calculate_double_precision():
        print("Double precision")
        min = -1022
        max = 1023
        t = 52
        print("Real min: ", 2 ** min)
        print("Real max: ", (2 - 2 ** (-t)) * 2 ** max)
        print("Epsilon value: ", (2 ** (-t)), "\n")

    #   Per le altre precisioni vale lo stesso concetto precedente solo che invece dei 64 bit iniziali adesso ne
    #   abbiamo 32, quindi:
    #   32 - 1 = 31, dove 1 è il segno
    #   31 - 8 = 23, dove 8 è la caratteristica, o esponente, e 23 è la mantissa.
    #   2^8 = 256 - 1, sono gli esponenti rappresentabili: 255
    #   255 - 2 = 253, dove i 2 sono gli estremi che verranno usati per altro
    #   (255 / 2) = 126 divisione intera, questo sarà l'esponente del numero minimo rappresentabile
    #   255 - 126 = 127 il quale sarà l'esponente del numero massimo rappresentabile.
    @staticmethod
    def calculate_single_precision():
        print("Single precision")
        min = -126
        max = 127
        t = 23
        print("Real min: ", 2 ** min)
        print("Real max: ", (2 - 2 ** (-t)) * 2 ** max)
        print("Epsilon value: ", (2 ** (-t)), "\n")

    #   Anche in questo metodo valgono tutte le considerazioni dei due metodi precedenti, calcolando però il tutto su
    #   16 bit totali.
    @staticmethod
    def calculate_half_precision():
        print("Half precision")
        min = -14
        max = 15
        t = 10
        print("Real min: ", 2 ** min)
        print("Real max: ", (2 - 2 ** (-t)) * 2 ** max)
        print("Epsilon value: ", (2 ** (-t)), "\n")
