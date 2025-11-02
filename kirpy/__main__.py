from . import main
import pstats


def f8_alt(x):
    return "%14.9f" % x

pstats.f8 = f8_alt

main.main(main.parse_args())